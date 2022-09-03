package ai.fedml.edge.request.listener;

import ai.fedml.edge.request.response.UserInfoResponse;

public interface OnUserInfoListener {
    void onGetUserInfo(UserInfoResponse.UserInfo userInfo);
}
