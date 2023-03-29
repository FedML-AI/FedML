import { ref } from 'vue';
import * as tf from '@tensorflow/tfjs';
const pred = ref(null);
export const useTestModel = function (canvasEl, model) {
  const canvas = canvasEl;
  if (!canvas) return console.error('useTestModel canvasEl can not be undefinded!');
  canvas.addEventListener('mousemove', (e) => {
    const { offsetX, offsetY } = e;
    if (e.buttons === 1) {
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = 'rgb(255,255,255)';
      ctx.fillRect(offsetX, offsetY, 15, 15);
    }
  });
  function clear() {
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'rgb(0,0,0)';
    ctx.fillRect(0, 0, canvasEl.width, canvasEl.height);
    pred.value = null;
  }
  clear();
  function predict() {
    if (!model) {
      return (pred.value = 'model not ready');
    }
    const input = tf.tidy(() => {
      return tf.image
        .resizeBilinear(tf.browser.fromPixels(canvas), [28, 28], true)
        .slice([0, 0, 0], [28, 28, 1])
        .toFloat()
        .div(255)
        .reshape([1, 28, 28, 1]);
    });
    pred.value = model.predict(input).argMax(1).dataSync()[0];
  }
  return { pred, predict, clear };
};
