package ai.fedml.edge.service.component;

import android.text.TextUtils;

import androidx.annotation.NonNull;

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;

import java.util.concurrent.ExecutionException;

import ai.fedml.edge.request.RequestManager;
import ai.fedml.edge.service.communicator.message.MessageDefine;
import ai.fedml.edge.utils.LogHelper;

public class TokenChecker implements MessageDefine {
    private final String mEdgeId;
    private final LoadingCache<String, String> tokenCache =
            CacheBuilder.newBuilder().maximumSize(2).build(new CacheLoader<String, String>() {
                @Override
                public String load(@NonNull String groupId) {
                    return RequestManager.getAccessToken(groupId, mEdgeId);
                }
            });

    public TokenChecker(final String edgeId) {
        mEdgeId = edgeId;
    }

    public boolean authentic(@NonNull final String groupId) {
        if (TextUtils.isEmpty(groupId)) {
            LogHelper.e("TokenChecker authentic failed: groupId is Empty.");
            return true;
        }
        // TODO: add authentic
        try {
            String accessToken = tokenCache.get(groupId);
            LogHelper.d("onStartTrain accessToken:%s ", accessToken);
            return true;
        } catch (ExecutionException e) {
            LogHelper.e(e, "tokenCache Execution failed.");
        }
        return false;
    }
}
