import argparse

from kfp.v2 import compiler
from kfp.v2 import dsl


@dsl.component(packages_to_install=["google-cloud-aiplatform"])
def batch_prediction_op(
        model_name: str, input_table: str, monitoring: bool, sample_uri: str, project_id: str, location: str) -> str:
    from google.cloud import aiplatform
    from google.cloud.aiplatform_v1beta1.services.job_service import JobServiceClient
    from google.cloud.aiplatform_v1beta1.types import (
        BatchDedicatedResources, BatchPredictionJob, BigQueryDestination, BigQuerySource,
        GcsSource, MachineSpec, ModelMonitoringAlertConfig, ModelMonitoringConfig,
        ModelMonitoringObjectiveConfig)

    aiplatform.init(project=project_id, location=location)
    model_monitoring_config = None
    if monitoring:
        skew_config = ModelMonitoringObjectiveConfig.TrainingPredictionSkewDetectionConfig()
        model_monitoring_config = ModelMonitoringConfig(
            alert_config=ModelMonitoringAlertConfig(
                enable_logging=True,
                email_alert_config=ModelMonitoringAlertConfig.EmailAlertConfig(
                    user_emails=[]
                )
            ),
            objective_configs=[
                ModelMonitoringObjectiveConfig(
                    training_dataset=ModelMonitoringObjectiveConfig.TrainingDataset(
                        data_format="csv",
                        gcs_source=GcsSource(uris=[sample_uri]),
                    ),
                    training_prediction_skew_detection_config=skew_config
                )
            ],
        )

    if not input_table.startswith("bq://"):
        input_table = f"bq://{input_table}"

    table_name_start_idx = input_table.rfind(".")
    output_dataset = input_table[:table_name_start_idx]

    matches = aiplatform.Model.list(filter=f"display_name={model_name}")
    model = matches[0].resource_name if matches else None

    batch_prediction_job = BatchPredictionJob(
        display_name=f"{model_name}-prediction-job",
        model=model,
        input_config=BatchPredictionJob.InputConfig(
            bigquery_source=BigQuerySource(input_uri=input_table),
            instances_format="bigquery"
        ),
        output_config=BatchPredictionJob.OutputConfig(
            bigquery_destination=BigQueryDestination(output_uri=output_dataset),
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
        project_id: str, location: str, model_name: str, source_table_id: str,
        monitoring: bool = False, training_sample_uri: str = "[none]",):
    batch_prediction_task = batch_prediction_op(
        model_name, source_table_id, monitoring, training_sample_uri, project_id, location)
    batch_prediction_task.set_display_name("batch-prediction")


def compile(filename: str):
    cmp = compiler.Compiler()
    cmp.compile(pipeline_func=batch_pipeline, package_path=filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-file-name", type=str, default="batch-pipeline.json")

    args = parser.parse_args()

    compile(args.pipeline_file_name)
