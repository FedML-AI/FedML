export abstract class Observer {
  abstract receive_message(msg_type: string | number, msg_params: unknown): any
}
