import { Arguments } from '../../arguments'
import { constant } from './constants'

export class CrossSiloLauncher {
  static launch_dist_trainers(_torch_client_filename: string, inputs: any) {
    const args = Arguments.load_arguments(constant.FEDML_TRAINING_PLATFORM_CROSS_SILO)
  }
}
