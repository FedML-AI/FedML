export abstract class Observer {
  abstract receive_message(msg_type, msg_params): null;
}
