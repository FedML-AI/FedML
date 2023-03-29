export class S3Storage {
  bucket_name;
  cn_region_name;
  cn_s3_sak;
  cn_s3_aki;

  constructor(s3_config_path) {
    this.bucket_name = null;
    this.cn_region_name = null;
    this.cn_s3_sak = null;
    this.cn_s3_aki = null;
    this.set_config_from_file(s3_config_path);
    this.set_config_from_objects(s3_config_path);
  }

  set_config_from_file(){

  }

  set_config_from_onjects(){
      
  }
}
