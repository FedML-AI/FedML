declare let AWS: any;
// import * as jpickle from 'jpickle';
// import unpickle from 'unpickle';

export class S3Storage {
  region;
  bucketName;
  accesskeyId;
  secretAccesskey;
  s3;
  constructor(s3_config) {
    this.region = s3_config.CN_REGION_NAME;
    this.bucketName = s3_config.BUCKET_NAME;
    this.accesskeyId = s3_config.CN_S3_AKI;
    this.secretAccesskey = s3_config.CN_S3_SAK;
    this.createS3Client(this.region, this.accesskeyId, this.secretAccesskey);
  }

  createS3Client(region_name, access_key_id, secret_access_key) {
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
    });
  }

  async write_model(message_key, model) {
    const model_pkl = JSON.stringify(model);
    console.log('check send model: ', model);
    const params = {
      Body: model_pkl,
      Bucket: this.bucketName,
      Key: message_key,
      Expires: 180,
    };
    const model_url = await this.s3.getSignedUrl('putObject', params);
    this.s3
      .putObject(params)
      .on('success', function (response) {
        console.log('putObject_response ', response);
      })
      .send();
    console.log('generated model_url: ', model_url);
    return model_url;
  }

  async read_model(message_key, verbose = false) {
    return new Promise(async (resolve, reject) => {
      const params = {
        Bucket: this.bucketName,
        Key: message_key,
      };
      console.log('receive message_key: ', message_key);
      await this.s3
        .getObject(params, (err, data) => {
          if (err) {
            reject(err);
          }
          resolve(data);
        })
        .send();
    });
  }
}
