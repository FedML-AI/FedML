export const IMAGE_W = 28
export const IMAGE_H = 28
// file transform to dom element
export function fileToImg(file) {
  return new Promise((resolve) => {
    const reader = new FileReader()
    reader.readAsDataURL(file)
    reader.onload = (e) => {
      let _a
      const img = document.createElement('img')
      img.src = (_a = e.target) === null || _a === void 0 ? void 0 : _a.result
      img.onload = () => resolve({ el: img, src: img.src, name: file.name })
    }
  })
}
// batch files transform to img
export function filesToImgs(files) {
  const proms = files.map(file => fileToImg(file))
  return Promise.all(proms)
}
/**
 * @description generate uuid if the image need to be deleted
 */
export const guid = function () {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0
    const v = c == 'x' ? r : (r & 0x3) | 0x8
    return v.toString(16)
  })
}
