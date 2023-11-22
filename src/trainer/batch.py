import argparse

from kfp import compiler
from kfp import dsl

BASE_IMAGE = "python:3.10"


@dsl.component(base_image=BASE_IMAGE)
def data_preparation_op(src_table: str, prepared_table: dsl.Output[dsl.Artifact]):
    # this would be the place to do any data preparation, we're just passing the input as output
    prepared_table.uri = src_table if src_table.startswith("bq://") else f"bq://{src_table}"


@dsl.component(base_image=BASE_IMAGE, packages_to_install=["google-cloud-aiplatform", "google-cloud-monitoring"])
def batch_prediction_op(
        model_name: str, input_table: dsl.Input[dsl.Artifact], monitoring_sample_uri: str,
        project_id: str, location: str, output_table: dsl.Output[dsl.Artifact]) -> str:
    import re

    from datetime import datetime

    from google.cloud import aiplatform
    from google.cloud import monitoring_v3
    from google.cloud.aiplatform_v1beta1.services.job_service import JobServiceClient
    from google.cloud.aiplatform_v1beta1.types import (
        BatchDedicatedResources, BatchPredictionJob, BigQueryDestination, BigQuerySource,
        GcsSource, MachineSpec, ModelMonitoringAlertConfig, ModelMonitoringConfig,
        ModelMonitoringObjectiveConfig)

    aiplatform.init(project=project_id, location=location)

    model_monitoring_config = None
    if monitoring_sample_uri != "[none]":
        monitoring_client = monitoring_v3.NotificationChannelServiceClient()
        notification_channel_req = monitoring_v3.ListNotificationChannelsRequest(
            name=f"projects/{project_id}", filter="type='pubsub'")
        notification_channel_res = monitoring_client.list_notification_channels(notification_channel_req)
        notification_channel_page = next(notification_channel_res.pages, None)
        notification_channels = []
        if notification_channel_page and len(notification_channel_page.notification_channels) > 0:
            notification_channels = [notification_channel_page.notification_channels[0].name]

        if monitoring_sample_uri.startswith("bq://"):
            training_dataset = ModelMonitoringObjectiveConfig.TrainingDataset(
                data_format="bigquery",
                gcs_source=BigQuerySource(input_uri=[monitoring_sample_uri])
            )
        else:
            if not monitoring_sample_uri.startswith("gs://"):
                monitoring_sample_uri = f"gs://{monitoring_sample_uri}"
            training_dataset = ModelMonitoringObjectiveConfig.TrainingDataset(
                data_format="csv",
                gcs_source=GcsSource(uris=[monitoring_sample_uri])
            )
        skew_config = ModelMonitoringObjectiveConfig.TrainingPredictionSkewDetectionConfig()
        model_monitoring_config = ModelMonitoringConfig(
            alert_config=ModelMonitoringAlertConfig(
                enable_logging=True,
                notification_channels=notification_channels,
                email_alert_config=ModelMonitoringAlertConfig.EmailAlertConfig(
                    user_emails=[]
                )
            ),
            objective_configs=[
                ModelMonitoringObjectiveConfig(
                    training_dataset=training_dataset,
                    training_prediction_skew_detection_config=skew_config
                )
            ],
        )

    table_name_start_idx = input_table.uri.rfind(".")
    output_dataset = input_table.uri[:table_name_start_idx]
    output_table_suffix = re.sub("[^a-zA-Z0-9_]+", "_", datetime.now().isoformat())
    output_table.uri = f"{output_dataset}.predictions_{output_table_suffix}"

    matches = aiplatform.Model.list(filter=f"display_name={model_name}")
    model = matches[0].resource_name if matches else None

    batch_prediction_job = BatchPredictionJob(
        display_name=f"{model_name}-prediction-job",
        model=model,
        input_config=BatchPredictionJob.InputConfig(
            bigquery_source=BigQuerySource(input_uri=input_table.uri),
            instances_format="bigquery"
        ),
        output_config=BatchPredictionJob.OutputConfig(
            bigquery_destination=BigQueryDestination(output_uri=output_table.uri),
            predictions_format="bigquery"
        ),
        dedicated_resources=BatchDedicatedResources(
            machine_spec=MachineSpec(machine_type="n1-standard-4"),
            starting_replica_count=1,
            max_replica_count=1,
        ),
        model_monitoring_config=model_monitoring_config
    )

    client = JobServiceClient(client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"})
    out = client.create_batch_prediction_job(
        parent=f"projects/{project_id}/locations/{location}",
        batch_prediction_job=batch_prediction_job,
    )

    return out.name.split("/")[-1]


@dsl.pipeline(name="taxi-tips-predictions")
def batch_pipeline(
        project_id: str, location: str, model_name: str, source_table_uri: str, training_sample_uri: str = "[none]"):

    data_preparation_task = data_preparation_op(src_table=source_table_uri).set_display_name("data-preparation")

    batch_prediction_task = batch_prediction_op(
        model_name=model_name, input_table=data_preparation_task.output,
        monitoring_sample_uri=training_sample_uri, project_id=project_id, location=location)
    batch_prediction_task.set_display_name("batch-prediction")


def compile(filename: str):
    cmp = compiler.Compiler()
    cmp.compile(pipeline_func=batch_pipeline, package_path=filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-file-name", type=str, default="batch-pipeline.json")

    args = parser.parse_args()

    compile(args.pipeline_file_name)
