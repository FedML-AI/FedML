// declare let AWS: any
// import * as jpickle from 'jpickle';
// import unpickle from 'unpickle';
import { S3 } from '@aws-sdk/client-s3'

export class S3Storage {
  region
  bucketName
  accesskeyId
  secretAccesskey
  s3?: S3
  constructor(s3_config: {
    CN_REGION_NAME: string
    BUCKET_NAME: string
    CN_S3_AKI: string
    CN_S3_SAK: string
  }) {
    this.region = s3_config.CN_REGION_NAME
    this.bucketName = s3_config.BUCKET_NAME
    this.accesskeyId = s3_config.CN_S3_AKI
    this.secretAccesskey = s3_config.CN_S3_SAK
    this.createS3Client(this.region, this.accesskeyId, this.secretAccesskey)
  }

  createS3Client(region_name: string, access_key_id: string, secret_access_key: string) {
    // aws.config.update({
    //   region: region_name,
    //   accessKeyId: access_key_id,
    //   secretAccessKey: secret_access_key,
    // });
    this.s3 = new S3({
      region: region_name,
      accessKeyId: access_key_id,
      secretAccessKey: secret_access_key,
      apiVersion: '2006-03-01',
    })
  }

  async write_model(message_key: string, model: object) {
    if (!this.s3)
      throw new Error('AWS S3 is not initialized')

    const model_pkl = JSON.stringify(model)
    console.log('check send model: ', model)
    const params = {
      Body: model_pkl,
      Bucket: this.bucketName,
      Key: message_key,
      Expires: 180,
    }
    const model_url = await this.s3.getSignedUrl('putObject', params)
    await this.s3
      .putObject(params)
      .then((response) => {
        console.log('putObject_response ', response)
      })
      // .on('success', (response) => {
      // })
      // .send()
    console.log('generated model_url: ', model_url)
    return model_url
  }

  async read_model(message_key: string, verbose = false) {
    if (!this.s3)
      throw new Error('AWS S3 is not initialized')

    const params = {
      Bucket: this.bucketName,
      Key: message_key,
    }
    return await this.s3.getObject(params)
  }
}
