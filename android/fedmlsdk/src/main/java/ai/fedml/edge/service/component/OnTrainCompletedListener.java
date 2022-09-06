package ai.fedml.edge.service.component;

/**
 * Training Completed Listener
 * @author joyerf
 */
public interface OnTrainCompletedListener {
    void onTrainCompleted(final String modelPath, final long edgeId, final int clientIdx, final long trainSamples);
}
