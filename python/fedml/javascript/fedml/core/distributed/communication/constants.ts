export class CommunicationConstants {
  static MSG_TYPE_CONNECTION_IS_READY = 0;
  static MSG_CLIENT_STATUS_OFFLINE = 'OFFLINE';
  static MSG_CLIENT_STATUS_IDLE = 'IDLE';
  static CLIENT_TOP_LAST_WILL_MSG = '/flclient/last_will_msg';
  static CLIENT_TOP_ACTIVE_MSG = '/flclient/active';
  static SERVER_TOP_LAST_WILL_MSG = '/flserver/last_will_msg';
  static SERVER_TOP_ACTIVE_MSG = '/flserver/active';
  static GRPC_BASE_PORT = 8890;
}
