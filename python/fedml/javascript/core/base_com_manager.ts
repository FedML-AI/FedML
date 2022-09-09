
import {Message} from './message';

export abstract class BaseCommunicationManager {
    abstract send_message(msg: Message);

    abstract add_observer(observer: Observer);

    abstract remove_observer(observer: Observer);

    abstract handle_receive_message();

    abstract stop_receive_message();
       
}