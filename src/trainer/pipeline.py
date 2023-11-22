import argparse

from kfp import compiler
from kfp import dsl

BASE_IMAGE = "python:3.10"


@dsl.component(base_image=BASE_IMAGE, packages_to_install=["google-cloud-bigquery"])
def data_extract_op(project_id: str, dataset: dsl.Output[dsl.Dataset]):
    import os

    from google.cloud import bigquery

    client = bigquery.Client()
    query = """
    EXPORT DATA OPTIONS(
        uri='{path}/*.csv',
        format='CSV',
        overwrite=true,
        header=true,
        field_delimiter=',') AS
    SELECT
        EXTRACT(MONTH from pickup_datetime) as trip_month,
        EXTRACT(DAY from pickup_datetime) as trip_day,
        EXTRACT(DAYOFWEEK from pickup_datetime) as trip_day_of_week,
        EXTRACT(HOUR from pickup_datetime) as trip_hour,
        TIMESTAMP_DIFF(dropoff_datetime, pickup_datetime, SECOND) as trip_duration,
        trip_distance,
        payment_type,
        pickup_location_id as pickup_zone,
        pickup_location_id as dropoff_zone,
        IF((SAFE_DIVIDE(tip_amount, fare_amount) >= 0.2), 1, 0) AS tip_bin
    FROM
        `bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_{year}` TABLESAMPLE SYSTEM (1 PERCENT)
    WHERE
        TIMESTAMP_DIFF(dropoff_datetime, pickup_datetime, SECOND) BETWEEN 300 AND 10800
    LIMIT {limit}
    """
    datasets = [
        (f"{dataset.path}/train", 2020, 10000),
        (f"{dataset.path}/val", 2020, 5000),
        (f"{dataset.path}/test", 2020, 1000)
    ]
    for ds in datasets:
        path = ds[0].replace("/gcs/", "gs://", 1)
        os.makedirs(path, exist_ok=True)
        # ignoring the provided location as this dataset is in US
        job = client.query(query.format(path=path, year=ds[1], limit=ds[2]), project=project_id, location="us")
        job.result()


@dsl.component(base_image=BASE_IMAGE)
def data_validation_op(dataset: dsl.Input[dsl.Dataset]) -> str:
    return "valid"


@dsl.component(base_image=BASE_IMAGE)
def data_preparation_op():
    pass


@dsl.component(base_image=BASE_IMAGE, packages_to_install=["google-cloud-aiplatform"])
def model_training_op(
        dataset: dsl.Input[dsl.Dataset],
        python_pkg: str,
        project_id: str,
        location: str,
        model: dsl.Output[dsl.Model]):
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=location, staging_bucket=f"gs://{project_id}/staging")

    pkg_with_uri = python_pkg if python_pkg.startswith("gs://") else f"gs://{project_id}/code/{python_pkg}"
    job = aiplatform.CustomPythonPackageTrainingJob(
        display_name="taxi-tips-custom-job",
        python_package_gcs_uri=pkg_with_uri if pkg_with_uri.endswith(".tar.gz") else f"{pkg_with_uri}.tar.gz",
        python_module_name="trainer.task",
        container_uri="us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-0:latest")

    job.run(
        replica_count=1,
        machine_type="n1-standard-4",
        args=[
            "--training-data-dir", f"{dataset.path}/train",
            "--validation-data-dir", f"{dataset.path}/val",
            "--output-dir", f"{model.path}"
        ]
        # accelerator_type=aiplatform.AcceleratorType.NVIDIA_TESLA_K80,
        # accelerator_count=4
    )


@dsl.component(base_image=BASE_IMAGE)
def model_evaluation_op(model: dsl.Input[dsl.Model], metrics: dsl.Output[dsl.ClassificationMetrics]):
    import json

    with open(f"{model.path}/metrics.json", "r") as f:
        model_metrics = json.load(f)

    conf_matrix = model_metrics["confusion_matrix"]
    metrics.log_confusion_matrix(categories=conf_matrix["categories"], matrix=conf_matrix["matrix"])

    curve = model_metrics["roc_curve"]
    metrics.log_roc_curve(fpr=curve["fpr"], tpr=curve["tpr"], threshold=curve["thresholds"])

    metrics.metadata["auc"] = model_metrics["auc"]


@dsl.component(base_image=BASE_IMAGE, packages_to_install=["google-cloud-aiplatform"])
def model_validation_op(
        model_name: str,
        metrics: dsl.Input[dsl.ClassificationMetrics],
        project_id: str,
        location: str) -> str:
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=location)
    matches = aiplatform.Model.list(filter=f"display_name={model_name}")
    if not matches:
        return "valid"
    else:
        latest_model_evaluation = matches[0].get_model_evaluation()
        return "valid" if metrics.metadata["auc"] > latest_model_evaluation.metrics["auRoc"] else "invalid"


@dsl.component(base_image=BASE_IMAGE, packages_to_install=["google-cloud-aiplatform"])
def model_upload_op(
        model: dsl.Input[dsl.Model],
        serving_container_image_uri: str,
        project_id: str,
        location: str,
        model_name: str) -> str:
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=location)
    matches = aiplatform.Model.list(filter=f"display_name={model_name}")
    parent_model = matches[0].resource_name if matches else None

    registered_model = aiplatform.Model.upload(
        display_name=model_name,
        parent_model=parent_model,
        artifact_uri=model.uri,
        serving_container_image_uri=serving_container_image_uri
    )

    return registered_model.versioned_resource_name


@dsl.component(base_image=BASE_IMAGE, packages_to_install=["google-cloud-aiplatform"])
def model_evaluation_upload_op(
        metrics: dsl.Input[dsl.ClassificationMetrics],
        model_resource_name: str,
        project_id: str,
        location: str):
    from google.api_core import gapic_v1
    from google.cloud import aiplatform
    from google.protobuf.struct_pb2 import Struct
    from google.protobuf.struct_pb2 import Value

    model_evaluation = {
        "display_name": "pipeline-eval",
        "metrics": Value(struct_value=Struct(fields={"auRoc": Value(number_value=metrics.metadata["auc"])})),
        "metrics_schema_uri": "gs://google-cloud-aiplatform/schema/modelevaluation/classification_metrics_1.0.0.yaml"
    }

    aiplatform.init(project=project_id, location=location)
    api_endpoint = location + '-aiplatform.googleapis.com'
    client = aiplatform.gapic.ModelServiceClient(client_info=gapic_v1.client_info.ClientInfo(
          user_agent="google-cloud-pipeline-components"),
      client_options={
          "api_endpoint": api_endpoint,
      })
    client.import_model_evaluation(parent=model_resource_name, model_evaluation=model_evaluation)


@dsl.component(base_image=BASE_IMAGE, packages_to_install=["google-cloud-aiplatform"])
def model_deployment_op(model_name: str, endpoint_name: str, project_id: str, location: str):
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=location)
    endpoints = aiplatform.Endpoint.list(filter=f"display_name={endpoint_name}")
    if endpoints:
        endpoint = endpoints[0]
    else:
        endpoint = aiplatform.Endpoint.create(display_name=endpoint_name, project=project_id, location=location)

    models = aiplatform.Model.list(filter=f"display_name={model_name}")
    if models:
        models[0].deploy(
            endpoint=endpoint,
            traffic_percentage=100,
            machine_type="n1-standard-2",
            min_replica_count=1,
            max_replica_count=4)


@dsl.component(base_image=BASE_IMAGE, packages_to_install=["google-cloud-aiplatform", "google-cloud-monitoring"])
def model_monitoring_op(
        dataset: dsl.Input[dsl.Dataset],
        monitoring_job_name: str,
        endpoint_name: str,
        project_id: str,
        location: str):

    from google.cloud import aiplatform
    from google.cloud import monitoring_v3
    from google.cloud.aiplatform_v1beta1.services.job_service import JobServiceClient
    from google.cloud.aiplatform_v1beta1.types import (
        GcsSource, ModelDeploymentMonitoringJob, ModelDeploymentMonitoringScheduleConfig, ModelMonitoringAlertConfig,
        ModelMonitoringObjectiveConfig, ModelDeploymentMonitoringObjectiveConfig, SamplingStrategy, job_state)
    aiplatform.init(project=project_id, location=location)

    endpoints = aiplatform.Endpoint.list(filter=f"display_name={endpoint_name}")
    if len(endpoints) > 0:
        monitoring_job_resource_name = endpoints[0].gca_resource.model_deployment_monitoring_job
        if (monitoring_job_resource_name):
            # can't update an existing monitoring job if it's pending so deleting it first
            job = aiplatform.ModelDeploymentMonitoringJob(monitoring_job_resource_name)
            if (job.state.value == job_state.JobState.JOB_STATE_RUNNING.value):
                # if job is running it needs to be paused first
                job.pause()
            job.delete()

        monitoring_client = monitoring_v3.NotificationChannelServiceClient()
        notification_channel_req = monitoring_v3.ListNotificationChannelsRequest(
            name=f"projects/{project_id}", filter="type='pubsub'")
        notification_channel_res = monitoring_client.list_notification_channels(notification_channel_req)
        notification_channel_page = next(notification_channel_res.pages, None)
        notification_channels = []
        if notification_channel_page and len(notification_channel_page.notification_channels) > 0:
            notification_channels = [notification_channel_page.notification_channels[0].name]

        skew_config = ModelMonitoringObjectiveConfig.TrainingPredictionSkewDetectionConfig()
        objective_config = ModelDeploymentMonitoringObjectiveConfig(
            deployed_model_id=endpoints[0].gca_resource.deployed_models[0].id,  # last deployed one is the first in list
            objective_config=ModelMonitoringObjectiveConfig(
                training_dataset=ModelMonitoringObjectiveConfig.TrainingDataset(
                    gcs_source=GcsSource(uris=[f"{dataset.path}/test/000000000000.csv".replace("/gcs/", "gs://", 1)]),
                    data_format="csv",
                    target_field="tip_bin"
                ),
                training_prediction_skew_detection_config=skew_config
            )
        )
        schedule_config = ModelDeploymentMonitoringScheduleConfig(
            monitor_interval={"seconds": 3600},
        )
        logging_strategy = SamplingStrategy(
            random_sample_config=SamplingStrategy.RandomSampleConfig(sample_rate=0.1)
        )
        alert_config = ModelMonitoringAlertConfig(
            enable_logging=True,
            notification_channels=notification_channels,
            email_alert_config=ModelMonitoringAlertConfig.EmailAlertConfig(
                user_emails=[]
            )
        )

        model_deployment_monitoring_job = ModelDeploymentMonitoringJob(
            display_name=monitoring_job_name,
            endpoint=endpoints[0].resource_name,
            model_deployment_monitoring_objective_configs=[objective_config],
            model_deployment_monitoring_schedule_config=schedule_config,
            logging_sampling_strategy=logging_strategy,
            model_monitoring_alert_config=alert_config
        )

        client = JobServiceClient(client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"})
        client.create_model_deployment_monitoring_job(
            parent=f"projects/{project_id}/locations/{location}",
            model_deployment_monitoring_job=model_deployment_monitoring_job
        )


@dsl.pipeline(name="taxi-tips-training")
def training_pipeline(
        project_id: str, location: str, python_pkg: str, endpoint: str = "[none]", monitoring_job: str = "[none]"):
    model_name = "taxi-tips"

    data_extraction_task = data_extract_op(
        project_id=project_id
    ).set_display_name("extract-data")

    data_validation_task = data_validation_op(
        dataset=data_extraction_task.outputs["dataset"]
    ).set_display_name("validate-data")

    data_preparation_task = data_preparation_op().set_display_name("prepare-data")
    data_preparation_task.after(data_validation_task)

    training_task = model_training_op(
        dataset=data_extraction_task.outputs["dataset"],
        python_pkg=python_pkg,
        project_id=project_id,
        location=location
    ).set_display_name("train-model")
    training_task.after(data_preparation_task)

    model_evaluation_task = model_evaluation_op(
        model=training_task.outputs["model"]
    ).set_display_name("evaluate-model")

    model_validation_task = model_validation_op(
        metrics=model_evaluation_task.outputs["metrics"],
        model_name=model_name,
        project_id=project_id,
        location=location
    ).set_display_name("validate-model")

    with dsl.If(model_validation_task.output == "valid", name="check-performance"):
        model_upload_task = model_upload_op(
            model=training_task.outputs["model"],
            model_name=model_name,
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-2:latest",
            project_id=project_id,
            location=location
        ).set_display_name("register-model")

        model_evaluation_upload_task = model_evaluation_upload_op(
            metrics=model_evaluation_task.outputs["metrics"],
            model_resource_name=model_upload_task.output,
            project_id=project_id,
            location=location
        ).set_display_name("register-model-evaluation")

        with dsl.If(endpoint != "[none]", name="check-if-endpoint-set"):
            model_deployment_task = model_deployment_op(
                model_name=model_name,
                endpoint_name=endpoint,
                project_id=project_id,
                location=location
            ).set_display_name("deploy-model")
            model_deployment_task.after(model_evaluation_upload_task)

            with dsl.If(monitoring_job != "[none]", name="check-if-monitoring-enabled"):
                model_monitoring_task = model_monitoring_op(
                    dataset=data_extraction_task.outputs["dataset"],
                    endpoint_name=endpoint,
                    monitoring_job_name=monitoring_job,
                    project_id=project_id,
                    location=location
                ).set_display_name("monitor-model")
                model_monitoring_task.after(model_deployment_task)


def compile(filename: str):
    cmp = compiler.Compiler()
    cmp.compile(pipeline_func=training_pipeline, package_path=filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-file-name", type=str, default="training-pipeline.json")

    args = parser.parse_args()

    compile(args.pipeline_file_name)
