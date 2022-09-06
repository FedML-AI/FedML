package ai.fedml.edge.service.component;

import android.content.Context;

import com.amazonaws.AmazonClientException;
import com.amazonaws.auth.AWSCredentials;
import com.amazonaws.auth.BasicAWSCredentials;
import com.amazonaws.mobileconnectors.s3.transferutility.TransferListener;
import com.amazonaws.mobileconnectors.s3.transferutility.TransferObserver;
import com.amazonaws.mobileconnectors.s3.transferutility.TransferUtility;
import com.amazonaws.regions.Region;
import com.amazonaws.services.s3.AmazonS3;
import com.amazonaws.services.s3.AmazonS3Client;
import com.amazonaws.services.s3.S3ClientOptions;
import com.amazonaws.services.s3.model.ObjectMetadata;
import com.amazonaws.services.s3.model.PutObjectResult;
import com.amazonaws.services.s3.model.S3Object;

import java.io.File;
import java.io.InputStream;

import ai.fedml.edge.request.response.ConfigResponse;
import ai.fedml.edge.service.ContextHolder;
import ai.fedml.edge.service.Initializer;
import ai.fedml.edge.utils.LogHelper;

public class RemoteStorage {
    public static final String TAG = "RemoteStorage";
    private final String ACCESS_KEY;
    private final String SECRET_KEY;
    private final String REGION_NAME;
    private final String BUCKET_UPLOAD;
    private final String BUCKET_DOWNLOAD;

    private AmazonS3 s3;
    private Context mAppContext;

    private final static class LazyHolder {
        private static final RemoteStorage sInstance = new RemoteStorage();
    }

    public static RemoteStorage getInstance() {
        return LazyHolder.sInstance;
    }

    private RemoteStorage() {
        ConfigResponse.S3Config s3Config = Initializer.getInstance().getS3Config();
        ACCESS_KEY = s3Config.getAk();
        SECRET_KEY = s3Config.getSk();
        REGION_NAME = s3Config.getRegionName();
        BUCKET_UPLOAD = s3Config.getBucket();
        BUCKET_DOWNLOAD = s3Config.getBucket();
        init(ContextHolder.getAppContext());
    }

    /**
     * initial  remote storage
     *
     * @param appContext Application Context
     */
    public void init(Context appContext) {
        mAppContext = appContext.getApplicationContext();
        Region region = Region.getRegion(REGION_NAME);
        AWSCredentials credentials = new BasicAWSCredentials(ACCESS_KEY, SECRET_KEY);
        S3ClientOptions clientOptions = S3ClientOptions.builder()
                .setAccelerateModeEnabled(false).build();
        s3 = new AmazonS3Client(credentials, region);
        s3.setS3ClientOptions(clientOptions);
        LogHelper.i("Initialized AmazonS3!");
    }

    /**
     * upload file
     *
     * @param key      object key
     * @param file     file
     * @param listener TransferListener
     * @return observer
     */
    public TransferObserver upload(String key, File file, TransferListener listener) {
        TransferUtility transferUtility = TransferUtility.builder().s3Client(s3).context(mAppContext).build();
        return transferUtility.upload(BUCKET_UPLOAD, key, file, new ObjectMetadata(), null, listener);
    }

    /**
     * download file
     *
     * @param key      The key under which the object to download is stored
     * @param file     The file to download the object's data to
     * @param listener a listener to attach to transfer observer
     * @return transfer observer
     */
    public TransferObserver download(String key, File file, TransferListener listener) {
        TransferUtility transferUtility = TransferUtility.builder().s3Client(s3).context(mAppContext).build();
        return transferUtility.download(BUCKET_DOWNLOAD, key, file, listener);
    }

    public InputStream readJson(String key) {
        S3Object s3Object = s3.getObject(BUCKET_DOWNLOAD, key);
        return s3Object.getObjectContent();
    }

    public void writeJson(String key, String json) {
        try {
            PutObjectResult result = s3.putObject(BUCKET_DOWNLOAD, key, json);
            String md5 = result.getContentMd5();
            LogHelper.d("writeJson(%s, %s)===%s", key, json, md5);
        } catch (AmazonClientException e) {
            LogHelper.e(e, "writeJson(%s, %s) Exception", key, json);
        }
    }
}
