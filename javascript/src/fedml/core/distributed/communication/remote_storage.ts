// declare let AWS: any
// import * as jpickle from 'jpickle';
// import unpickle from 'unpickle';
// import AWS from 'aws-sdk'
// @ts-ignore
import type { S3 } from 'aws-sdk'
import AWS from '../../../../libs/aws-sdk'

type GetObjectOutput = S3.GetObjectOutput

export class S3Storage {
  region
  bucketName
  accesskeyId
  secretAccesskey
  s3?: S3

  constructor(s3_config: { CN_REGION_NAME: any; BUCKET_NAME: any; CN_S3_AKI: any; CN_S3_SAK: any }) {
    this.region = s3_config.CN_REGION_NAME
    this.bucketName = s3_config.BUCKET_NAME
    this.accesskeyId = s3_config.CN_S3_AKI
    this.secretAccesskey = s3_config.CN_S3_SAK
    this.createS3Client(this.region, this.accesskeyId, this.secretAccesskey)
  }

  createS3Client(region_name: any, access_key_id: any, secret_access_key: any) {
    // aws.config.update({
    //   region: region_name,
    //   accessKeyId: access_key_id,
    //   secretAccessKey: secret_access_key,
    // });
    this.s3 = new AWS.S3({
      region: region_name,
      accessKeyId: access_key_id,
      secretAccessKey: secret_access_key,
      apiVersion: '2006-03-01',
    })
  }

  async write_model(message_key: string, model: any) {
    if (!this.s3)
      throw new Error('AWS S3 client is not established~')

    const model_pkl = JSON.stringify(model)
    console.log('check send model: ', model)
    const params = {
      Body: model_pkl,
      Bucket: this.bucketName,
      Key: message_key,
      Expires: 180,
    }
    const model_url = await this.s3.getSignedUrl('putObject', params)
    this.s3
    // @ts-ignore
      .putObject(params)
      .on('success', (response) => {
        console.log('putObject_response ', response)
      })
      .send()
    console.log('generated model_url: ', model_url)
    return model_url
  }

  async read_model(message_key: string) {
    return new Promise<GetObjectOutput | null>((resolve, reject) => {
      if (!this.s3)
        return reject(new Error('AWS S3 client is not established~'))

      const params = {
        Bucket: this.bucketName,
        Key: message_key,
      }
      console.log('receive message_key: ', message_key)
      this.s3.getObject(params, (err, data) => {
        if (err)
          reject(err)
        resolve(data)
      })
        .send()
    })
  }
}
