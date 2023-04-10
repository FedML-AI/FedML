import { MqttS3MultiClientsCommManager } from './mqtt_s3_multi_clients_comm_manager'

const mqtt_config = {
    BROKER_HOST: data.mqtt_config.BROKER_HOST,
    MQTT_PWD: data.mqtt_config.MQTT_PWD,
    BROKER_PORT: data.mqtt_config.BROKER_PORT,
    MQTT_KEEPALIVE: data.mqtt_config.MQTT_KEEPALIVE,
    MQTT_USER: data.mqtt_config.MQTT_USER,
  }
  const s3_config = {
    CN_S3_SAK: data.s3_config.CN_S3_SAK,
    CN_REGION_NAME: data.s3_config.CN_REGION_NAME,
    CN_S3_AKI: data.s3_config.CN_S3_AKI,
    BUCKET_NAME: data.s3_config.BUCKET_NAME,
  }
  this.com_manager = new MqttS3MultiClientsCommManager(
    mqtt_config,
    s3_config,
    this.args.run_id,
    this.rank,
    this.size,
    this.args,
  )