import type { Message } from './message';
import type { Observer } from './observer';

export abstract class BaseCommunicationManager {
  abstract send_message(msg: Message): any;

  abstract add_observer(observer: Observer): any;

  abstract remove_observer(observer: Observer): any;

  abstract handle_receive_message(): any;

  abstract stop_receive_message(): any;
}
