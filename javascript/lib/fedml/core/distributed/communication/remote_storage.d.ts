export declare class S3Storage {
    region: any;
    bucketName: any;
    accesskeyId: any;
    secretAccesskey: any;
    s3: any;
    constructor(s3_config: any);
    createS3Client(region_name: any, access_key_id: any, secret_access_key: any): void;
    write_model(message_key: any, model: any): Promise<any>;
    read_model(message_key: any, verbose?: boolean): Promise<any>;
}
