<template>
  <div class="train-box">
    <div class="train-progress-box">
      <canvas
        ref="preCanvas"
        width="280"
        height="280"
        style="width: 280px; height: 280px; margin: 10px 0"
      ></canvas>
      <p v-if="testModel" style="margin-left:80px;">
        <a-button type="dashed" @click="testModel.clear">clear</a-button>
        <a-button type="dashed" @click="testModel.predict">predict</a-button>
      </p>
      <h3 style="margin-left:100px;">predict result:{{ testModel?.pred }}</h3>
    </div>
    <div class="oprate-box">
      <a-button type="dashed" @click="showTrainModal = true">training model</a-button>
      <a-button type="dashed" @click="saveModel">export training model</a-button>
    </div>
    <!-- set the training data window -->
    <modal
      v-model:visible="showTrainModal"
      width="500px"
      height="600px"
      title="load training data"
      @ok="handelTrainModalOk"
      @cancel="clearModalInput"
    >
      <a-spin :spinning="loadDataLoading">
        <div class="load-data-box">
          <!-- select the image button -->
          <input
            ref="imgInputRef"
            type="file"
            :value="imgInptValue"
            accept="image/png,image/jpg,image/jpeg"
            @change="handleFileChange"
            multiple
            style="display: none"
          />
          <a-button style="margin-left: 8px" @click="handleUploadBtnClick">
            <upload-outlined></upload-outlined>
            upload
          </a-button>
          <div style="display: inline-block; margin-left: 8px">Label 0</div>
          <!-- select whether select the same label -->
          <div class="slect-fill-all">
            <span class="text" style="margin-left: 8px">whether fill the same Label: </span>
            <a-switch v-model:checked="fillAllLabel" />
            <a-input
              type="number"
              v-if="fillAllLabel"
              v-model:value="fillAllLabelValue"
              placeholder="label"
            />
          </div>
          <ul>
            <li v-for="(item, idx) in imgList" :key="item.uid">
              <img :src="item.src" alt="" />
              <div class="label" v-if="!fillAllLabel">
                <span>Label:</span>
                <a-input type="number" v-model:value="item.label" placeholder="label" />
              </div>
            </li>
          </ul>
        </div>
        <div class="load-data-box">
          <!-- select the image button -->
          <input
            ref="imgInputRef1"
            type="file"
            :value="imgInptValue1"
            accept="image/png,image/jpg,image/jpeg"
            @change="handleFileChange1"
            multiple
            style="display: none"
          />
          <a-button style="margin-left: 8px" @click="handleUploadBtnClick1">
            <upload-outlined />
            upload
          </a-button>
          <div style="display: inline-block; margin-left: 8px">Label 1</div>
          <!-- select whether select the same label -->
          <div class="slect-fill-all">
            <span class="text" style="margin-left: 8px">whether fill the same Label: </span>
            <a-switch v-model:checked="fillAllLabel1" />
            <a-input
              type="number"
              v-if="fillAllLabel1"
              v-model:value="fillAllLabelValue1"
              placeholder="label"
            />
          </div>
          <ul>
            <li v-for="(item, idx) in imgList1" :key="item.uid">
              <img :src="item.src" alt="" />
              <div class="label" v-if="!fillAllLabel1">
                <span>Label:</span>
                <a-input type="number" v-model:value="item.label" placeholder="label" />
              </div>
            </li>
          </ul>
        </div>
        <div class="load-data-box">
          <!-- select the image button -->
          <input
            ref="imgInputRef2"
            type="file"
            :value="imgInptValue2"
            accept="image/png,image/jpg,image/jpeg"
            @change="handleFileChange2"
            multiple
            style="display: none"
          />
          <a-button style="margin-left: 8px" @click="handleUploadBtnClick2">
            <upload-outlined />
            upload
          </a-button>
          <div style="display: inline-block; margin-left: 8px">Label 2</div>
          <!-- select whether select the same label -->
          <div class="slect-fill-all">
            <span class="text" style="margin-left: 8px">whether fill the same Label: </span>
            <a-switch v-model:checked="fillAllLabel2" />
            <a-input
              type="number"
              v-if="fillAllLabel2"
              v-model:value="fillAllLabelValue2"
              placeholder="label"
            />
          </div>
          <ul>
            <li v-for="(item, idx) in imgList2" :key="item.uid">
              <img :src="item.src" alt="" />
              <div class="label" v-if="!fillAllLabel2">
                <span>Label:</span>
                <a-input type="number" v-model:value="item.label" placeholder="label" />
              </div>
            </li>
          </ul>
        </div>
        <div class="load-data-box">
          <!-- select the image button -->
          <input
            ref="imgInputRef3"
            type="file"
            :value="imgInptValue3"
            accept="image/png,image/jpg,image/jpeg"
            @change="handleFileChange3"
            multiple
            style="display: none"
          />
          <a-button style="margin-left: 8px" @click="handleUploadBtnClick3">
            <upload-outlined />
            upload
          </a-button>
          <div style="display: inline-block; margin-left: 8px">Label 3</div>
          <!-- select whether select the same label -->
          <div class="slect-fill-all">
            <span class="text" style="margin-left: 8px">whether fill the same Label: </span>
            <a-switch v-model:checked="fillAllLabel3" />
            <a-input
              type="number"
              v-if="fillAllLabel3"
              v-model:value="fillAllLabelValue3"
              placeholder="label"
            />
          </div>
          <ul>
            <li v-for="(item, idx) in imgList3" :key="item.uid">
              <img :src="item.src" alt="" />
              <div class="label" v-if="!fillAllLabel3">
                <span>Label:</span>
                <a-input type="number" v-model:value="item.label" placeholder="label" />
              </div>
            </li>
          </ul>
        </div>
        <div class="load-data-box">
          <!-- select the image button -->
          <input
            ref="imgInputRef4"
            type="file"
            :value="imgInptValue4"
            accept="image/png,image/jpg,image/jpeg"
            @change="handleFileChange4"
            multiple
            style="display: none"
          />
          <a-button style="margin-left: 8px" @click="handleUploadBtnClick4">
            <upload-outlined />
            upload
          </a-button>
          <div style="display: inline-block; margin-left: 8px">Label 4</div>
          <!-- select whether select the same label -->
          <div class="slect-fill-all">
            <span class="text" style="margin-left: 8px">whether fill the same Label: </span>
            <a-switch v-model:checked="fillAllLabel4" />
            <a-input
              type="number"
              v-if="fillAllLabel4"
              v-model:value="fillAllLabelValue4"
              placeholder="label"
            />
          </div>
          <ul>
            <li v-for="(item, idx) in imgList4" :key="item.uid">
              <img :src="item.src" alt="" />
              <div class="label" v-if="!fillAllLabel4">
                <span>Label:</span>
                <a-input type="number" v-model:value="item.label" placeholder="label" />
              </div>
            </li>
          </ul>
        </div>
        <div class="load-data-box">
          <!-- select the image button -->
          <input
            ref="imgInputRef5"
            type="file"
            :value="imgInptValue5"
            accept="image/png,image/jpg,image/jpeg"
            @change="handleFileChange5"
            multiple
            style="display: none"
          />
          <a-button style="margin-left: 8px" @click="handleUploadBtnClick5">
            <upload-outlined />
            upload
          </a-button>
          <div style="display: inline-block; margin-left: 8px">Label 5</div>
          <!-- select whether select the same label -->
          <div class="slect-fill-all">
            <span class="text" style="margin-left: 8px">whether fill the same Label: </span>
            <a-switch v-model:checked="fillAllLabel5" />
            <a-input
              type="number"
              v-if="fillAllLabel5"
              v-model:value="fillAllLabelValue5"
              placeholder="label"
            />
          </div>
          <ul>
            <li v-for="(item, idx) in imgList5" :key="item.uid">
              <img :src="item.src" alt="" />
              <div class="label" v-if="!fillAllLabel5">
                <span>Label:</span>
                <a-input type="number" v-model:value="item.label" placeholder="label" />
              </div>
            </li>
          </ul>
        </div>
        <div class="load-data-box">
          <!-- select the image button -->
          <input
            ref="imgInputRef6"
            type="file"
            :value="imgInptValue6"
            accept="image/png,image/jpg,image/jpeg"
            @change="handleFileChange6"
            multiple
            style="display: none"
          />
          <a-button style="margin-left: 8px" @click="handleUploadBtnClick6">
            <upload-outlined />
            upload
          </a-button>
          <div style="display: inline-block; margin-left: 8px">Label 6</div>
          <!-- select whether select the same label -->
          <div class="slect-fill-all">
            <span class="text" style="margin-left: 8px">whether fill the same Label: </span>
            <a-switch v-model:checked="fillAllLabel6" />
            <a-input
              type="number"
              v-if="fillAllLabel6"
              v-model:value="fillAllLabelValue6"
              placeholder="label"
            />
          </div>
          <ul>
            <li v-for="(item, idx) in imgList6" :key="item.uid">
              <img :src="item.src" alt="" />
              <div class="label" v-if="!fillAllLabel6">
                <span>Label:</span>
                <a-input type="number" v-model:value="item.label" placeholder="label" />
              </div>
            </li>
          </ul>
        </div>
        <div class="load-data-box">
          <!-- select the image button -->
          <input
            ref="imgInputRef7"
            type="file"
            :value="imgInptValue7"
            accept="image/png,image/jpg,image/jpeg"
            @change="handleFileChange7"
            multiple
            style="display: none"
          />
          <a-button style="margin-left: 8px" @click="handleUploadBtnClick7">
            <upload-outlined />
            upload
          </a-button>
          <div style="display: inline-block; margin-left: 8px">Label 7</div>
          <!-- select whether select the same label -->
          <div class="slect-fill-all">
            <span class="text" style="margin-left: 8px">whether fill the same Label: </span>
            <a-switch v-model:checked="fillAllLabel7" />
            <a-input
              type="number"
              v-if="fillAllLabel7"
              v-model:value="fillAllLabelValue7"
              placeholder="label"
            />
          </div>
          <ul>
            <li v-for="(item, idx) in imgList7" :key="item.uid">
              <img :src="item.src" alt="" />
              <div class="label" v-if="!fillAllLabel7">
                <span>Label:</span>
                <a-input type="number" v-model:value="item.label" placeholder="label" />
              </div>
            </li>
          </ul>
        </div>
        <div class="load-data-box">
          <!-- select the image button -->
          <input
            ref="imgInputRef8"
            type="file"
            :value="imgInptValue8"
            accept="image/png,image/jpg,image/jpeg"
            @change="handleFileChange8"
            multiple
            style="display: none"
          />
          <a-button style="margin-left: 8px" @click="handleUploadBtnClick8">
            <upload-outlined />
            upload
          </a-button>
          <div style="display: inline-block; margin-left: 8px">Label 8</div>
          <!-- select whether select the same label -->
          <div class="slect-fill-all">
            <span class="text" style="margin-left: 8px">whether fill the same Label: </span>
            <a-switch v-model:checked="fillAllLabel8" />
            <a-input
              type="number"
              v-if="fillAllLabel8"
              v-model:value="fillAllLabelValue8"
              placeholder="label"
            />
          </div>
          <ul>
            <li v-for="(item, idx) in imgList8" :key="item.uid">
              <img :src="item.src" alt="" />
              <div class="label" v-if="!fillAllLabel8">
                <span>Label:</span>
                <a-input type="number" v-model:value="item.label" placeholder="label" />
              </div>
            </li>
          </ul>
        </div>
        <div class="load-data-box">
          <!-- select the image button -->
          <input
            ref="imgInputRef9"
            type="file"
            :value="imgInptValue9"
            accept="image/png,image/jpg,image/jpeg"
            @change="handleFileChange9"
            multiple
            style="display: none"
          />
          <a-button style="margin-left: 8px" @click="handleUploadBtnClick9">
            <upload-outlined />
            upload
          </a-button>
          <div style="display: inline-block; margin-left: 8px">Label 9</div>
          <!-- select whether select the same label -->
          <div class="slect-fill-all">
            <span class="text" style="margin-left: 8px">whether fill the same Label: </span>
            <a-switch v-model:checked="fillAllLabel9" />
            <a-input
              type="number"
              v-if="fillAllLabel9"
              v-model:value="fillAllLabelValue9"
              placeholder="label"
            />
          </div>
          <ul>
            <li v-for="(item, idx) in imgList9" :key="item.uid">
              <img :src="item.src" alt="" />
              <div class="label" v-if="!fillAllLabel9">
                <span>Label:</span>
                <a-input type="number" v-model:value="item.label" placeholder="label" />
              </div>
            </li>
          </ul>
        </div>
      </a-spin>
    </modal>
  </div>
</template>

<script lang="ts">
  export default { name: 'MnistTrainTask' };
</script>
<script setup lang="ts">
  import * as tf from '@tensorflow/tfjs';
  import { computed, ref, onMounted } from 'vue';
  import {
    Button as aButton,
    Modal,
    Input as aInput,
    Switch as aSwitch,
    message,
    Spin as aSpin,
  } from 'ant-design-vue';
  import { IMAGE_H, IMAGE_W, guid, filesToImgs } from './components/data_preprocess';
  import { useTestModel } from './components/useTestModel';
  import { mnist_trainer } from './components/mnist_trainer';
  import { LoadImgData } from './components/mnist_loadData';
  // import { runApplication } from '/@/api/appStore/index';
  import { fedml_train } from './components/mnist_fedml_train';
  const { model, saveModel, doTrain } = mnist_trainer();
  // canvas draw to test
  const testModel = ref(null);
  const preCanvas = ref(null);
  const showTrainModal = ref(false);
  const trainTestRatio = ref(4 / 5);
  // select the files load image to the format：{ el: img, src: img.src, name: file.name,label:xxx }
  const imgList = ref([]);
  const imgList1 = ref([]);
  const imgList2 = ref([]);
  const imgList3 = ref([]);
  const imgList4 = ref([]);
  const imgList5 = ref([]);
  const imgList6 = ref([]);
  const imgList7 = ref([]);
  const imgList8 = ref([]);
  const imgList9 = ref([]);
  // total training data
  const imgListTotal = ref([]);
  // total testing data
  const imgListTestTotal = ref([]);
  const imgInputRef = ref(null);
  const imgInputRef1 = ref(null);
  const imgInputRef2 = ref(null);
  const imgInputRef3 = ref(null);
  const imgInputRef4 = ref(null);
  const imgInputRef5 = ref(null);
  const imgInputRef6 = ref(null);
  const imgInputRef7 = ref(null);
  const imgInputRef8 = ref(null);
  const imgInputRef9 = ref(null);
  const imgInptValue = ref([]);
  const imgInptValue1 = ref([]);
  const imgInptValue2 = ref([]);
  const imgInptValue3 = ref([]);
  const imgInptValue4 = ref([]);
  const imgInptValue5 = ref([]);
  const imgInptValue6 = ref([]);
  const imgInptValue7 = ref([]);
  const imgInptValue8 = ref([]);
  const imgInptValue9 = ref([]);
  // whether fill the same Label
  const fillAllLabel = ref(false);
  const fillAllLabel1 = ref(false);
  const fillAllLabel2 = ref(false);
  const fillAllLabel3 = ref(false);
  const fillAllLabel4 = ref(false);
  const fillAllLabel5 = ref(false);
  const fillAllLabel6 = ref(false);
  const fillAllLabel7 = ref(false);
  const fillAllLabel8 = ref(false);
  const fillAllLabel9 = ref(false);
  // fill label value
  const fillAllLabelValue = ref('');
  const fillAllLabelValue1 = ref('');
  const fillAllLabelValue2 = ref('');
  const fillAllLabelValue3 = ref('');
  const fillAllLabelValue4 = ref('');
  const fillAllLabelValue5 = ref('');
  const fillAllLabelValue6 = ref('');
  const fillAllLabelValue7 = ref('');
  const fillAllLabelValue8 = ref('');
  const fillAllLabelValue9 = ref('');
  const loadDataLoading = ref(false);
  // load the data popup
  async function handelTrainModalOk() {
    // check the data
    if (fillAllLabel.value) {
      imgList.value.forEach((item) => {
        item.label = fillAllLabelValue.value;
      });
      const train_length_label0 = imgList.value.length * trainTestRatio.value;
      for (let i = train_length_label0; i < imgList.value.length; i++) {
        imgListTestTotal.value.push(imgList.value.at(i));
      }
      for (let j = 0; j < train_length_label0; j++) {
        imgListTotal.value.push(imgList.value.at(j));
      }
    }
    if (fillAllLabel1.value) {
      imgList1.value.forEach((item) => {
        item.label = fillAllLabelValue1.value;
      });
      const train_length_label1 = imgList1.value.length * trainTestRatio.value;
      for (let i = train_length_label1; i < imgList1.value.length; i++) {
        imgListTestTotal.value.push(imgList1.value.at(i));
      }
      for (let j = 0; j < train_length_label1; j++) {
        imgListTotal.value.push(imgList1.value.at(j));
      }
    }
    if (fillAllLabel2.value) {
      imgList2.value.forEach((item) => {
        item.label = fillAllLabelValue2.value;
      });
      const train_length_label2 = imgList2.value.length * trainTestRatio.value;
      for (let i = train_length_label2; i < imgList2.value.length; i++) {
        imgListTestTotal.value.push(imgList2.value.at(i));
      }
      for (let j = 0; j < train_length_label2; j++) {
        imgListTotal.value.push(imgList2.value.at(j));
      }
    }
    if (fillAllLabel3.value) {
      imgList3.value.forEach((item) => {
        item.label = fillAllLabelValue3.value;
      });
      const train_length_label3 = imgList3.value.length * trainTestRatio.value;
      for (let i = train_length_label3; i < imgList3.value.length; i++) {
        imgListTestTotal.value.push(imgList3.value.at(i));
      }
      for (let j = 0; j < train_length_label3; j++) {
        imgListTotal.value.push(imgList3.value.at(j));
      }
    }
    if (fillAllLabel4.value) {
      imgList4.value.forEach((item) => {
        item.label = fillAllLabelValue4.value;
      });
      const train_length_label4 = imgList4.value.length * trainTestRatio.value;
      for (let i = train_length_label4; i < imgList4.value.length; i++) {
        imgListTestTotal.value.push(imgList4.value.at(i));
      }
      for (let j = 0; j < train_length_label4; j++) {
        imgListTotal.value.push(imgList4.value.at(j));
      }
    }
    if (fillAllLabel5.value) {
      imgList5.value.forEach((item) => {
        item.label = fillAllLabelValue5.value;
      });
      const train_length_label5 = imgList5.value.length * trainTestRatio.value;
      for (let i = train_length_label5; i < imgList5.value.length; i++) {
        imgListTestTotal.value.push(imgList5.value.at(i));
      }
      for (let j = 0; j < train_length_label5; j++) {
        imgListTotal.value.push(imgList5.value.at(j));
      }
    }
    if (fillAllLabel6.value) {
      imgList6.value.forEach((item) => {
        item.label = fillAllLabelValue6.value;
      });
      const train_length_label6 = imgList6.value.length * trainTestRatio.value;
      for (let i = train_length_label6; i < imgList6.value.length; i++) {
        imgListTestTotal.value.push(imgList6.value.at(i));
      }
      for (let j = 0; j < train_length_label6; j++) {
        imgListTotal.value.push(imgList6.value.at(j));
      }
    }
    if (fillAllLabel7.value) {
      imgList7.value.forEach((item) => {
        item.label = fillAllLabelValue7.value;
      });
      const train_length_label7 = imgList7.value.length * trainTestRatio.value;
      for (let i = train_length_label7; i < imgList7.value.length; i++) {
        imgListTestTotal.value.push(imgList7.value.at(i));
      }
      for (let j = 0; j < train_length_label7; j++) {
        imgListTotal.value.push(imgList7.value.at(j));
      }
    }
    if (fillAllLabel8.value) {
      imgList8.value.forEach((item) => {
        item.label = fillAllLabelValue8.value;
      });
      const train_length_label8 = imgList8.value.length * trainTestRatio.value;
      for (let i = train_length_label8; i < imgList8.value.length; i++) {
        imgListTestTotal.value.push(imgList8.value.at(i));
      }
      for (let j = 0; j < train_length_label8; j++) {
        imgListTotal.value.push(imgList8.value.at(j));
      }
    }
    if (fillAllLabel9.value) {
      imgList9.value.forEach((item) => {
        item.label = fillAllLabelValue9.value;
      });
      const train_length_label9 = imgList9.value.length * trainTestRatio.value;
      for (let i = train_length_label9; i < imgList9.value.length; i++) {
        imgListTestTotal.value.push(imgList9.value.at(i));
      }
      for (let j = 0; j < train_length_label9; j++) {
        imgListTotal.value.push(imgList9.value.at(j));
      }
    }
    loadDataLoading.value = true;
    const { xs, ys } = await genTrainDataByImgList(imgListTotal);
    const { txs, tys } = await genTestDataByImgList(imgListTestTotal);
    clearModalInput();
    loadDataLoading.value = false;
    showTrainModal.value = false;
    const params = {
      applicationId: 189,
      devices: [
        {
          serverId: 726,
          edgeIds: [718],
          account: 303,
        },
      ],
      name: 'rear_ryan',
      projectId: '472',
      urls: [],
    };
    // await runApplication(params);

    doTrain(xs, ys, txs, tys);
  }

  // change the select the image data to tensor data {xs,ys}
  async function genTrainDataByImgList(imgList) {
    let xsTensProms = [];
    imgList.value.forEach((item) => {
      const imgLoader = new LoadImgData(item.src);
      xsTensProms.push(imgLoader.load());
    });
    let xsArr = await Promise.all(xsTensProms);
    const { xs, ys } = tf.tidy(() => {
      const xs = tf.concat(xsArr);
      let ysArr = [];
      imgList.value.map((item) => {
        let idx = parseInt(item.label);
        let label = new Array(10).fill(0);
        label[idx] = 1;
        ysArr.push(label);
      });
      const ys = tf.tensor(ysArr);
      return { xs, ys };
    });
    return { xs: xs.reshape([imgList.value.length, 28, 28, 1]), ys };
  }
  // change the select the image data to tensor data {xs,ys}
  async function genTestDataByImgList(imgList) {
    let xsTensProms = [];
    imgList.value.forEach((item) => {
      const imgLoader = new LoadImgData(item.src);
      xsTensProms.push(imgLoader.load());
    });
    let xsArr = await Promise.all(xsTensProms);
    const { xs, ys } = tf.tidy(() => {
      const xs = tf.concat(xsArr);
      let ysArr = [];
      imgList.value.map((item) => {
        let idx = parseInt(item.label);
        let label = new Array(10).fill(0);
        label[idx] = 1;
        ysArr.push(label);
      });
      const ys = tf.tensor2d(ysArr);
      return { xs, ys };
    });
    return { txs: xs.reshape([imgList.value.length, 28, 28, 1]), tys: ys };
  }
  async function handleUploadBtnClick() {
    await fedml_train();
    imgInputRef.value.click();
  }
  const handleUploadBtnClick1 = function () {
    imgInputRef1.value.click();
  };
  const handleUploadBtnClick2 = function () {
    imgInputRef2.value.click();
  };
  const handleUploadBtnClick3 = function () {
    imgInputRef3.value.click();
  };
  const handleUploadBtnClick4 = function () {
    imgInputRef4.value.click();
  };
  const handleUploadBtnClick5 = function () {
    imgInputRef5.value.click();
  };
  const handleUploadBtnClick6 = function () {
    imgInputRef6.value.click();
  };
  const handleUploadBtnClick7 = function () {
    imgInputRef7.value.click();
  };
  const handleUploadBtnClick8 = function () {
    imgInputRef8.value.click();
  };
  const handleUploadBtnClick9 = function () {
    imgInputRef9.value.click();
  };
  const handleFileChange = async function (e) {
    let files = e.target.files;
    let imgs = [];
    if (files.length) {
      imgs = await filesToImgs(Array.from(files));
    }
    imgs = imgs.filter((item) => {
      let checkSizeSucc = item.el.width == IMAGE_W && item.el.height == IMAGE_H;
      if (!checkSizeSucc) {
        message.info(`${item.name} image size unnormal[${IMAGE_W},${IMAGE_H}], neglected`);
      }
      return checkSizeSucc;
    });
    imgs.forEach((item) => {
      item.label = '';
      item.uid = guid();
    });
    imgList.value.push(...imgs);
    imgInptValue.value = [];
  };
  const handleFileChange1 = async function (e) {
    let files = e.target.files;
    let imgs = [];
    if (files.length) {
      imgs = await filesToImgs(Array.from(files));
    }
    imgs = imgs.filter((item) => {
      let checkSizeSucc = item.el.width == IMAGE_W && item.el.height == IMAGE_H;
      if (!checkSizeSucc) {
        message.info(`${item.name} image size unnormal[${IMAGE_W},${IMAGE_H}], neglected`);
      }
      return checkSizeSucc;
    });
    imgs.forEach((item) => {
      item.label = '';
      item.uid = guid();
    });
    imgList1.value.push(...imgs);
    imgInptValue1.value = [];
  };
  const handleFileChange2 = async function (e) {
    let files = e.target.files;
    let imgs = [];
    if (files.length) {
      imgs = await filesToImgs(Array.from(files));
    }
    imgs = imgs.filter((item) => {
      let checkSizeSucc = item.el.width == IMAGE_W && item.el.height == IMAGE_H;
      if (!checkSizeSucc) {
        message.info(`${item.name} image size unnormal[${IMAGE_W},${IMAGE_H}], neglected`);
      }
      return checkSizeSucc;
    });
    imgs.forEach((item) => {
      item.label = '';
      item.uid = guid();
    });
    imgList2.value.push(...imgs);
    imgInptValue2.value = [];
  };
  const handleFileChange3 = async function (e) {
    let files = e.target.files;
    let imgs = [];
    if (files.length) {
      imgs = await filesToImgs(Array.from(files));
    }
    imgs = imgs.filter((item) => {
      let checkSizeSucc = item.el.width == IMAGE_W && item.el.height == IMAGE_H;
      if (!checkSizeSucc) {
        message.info(`${item.name} image size unnormal[${IMAGE_W},${IMAGE_H}], neglected`);
      }
      return checkSizeSucc;
    });
    imgs.forEach((item) => {
      item.label = '';
      item.uid = guid();
    });
    imgList3.value.push(...imgs);
    imgInptValue3.value = [];
  };
  const handleFileChange4 = async function (e) {
    let files = e.target.files;
    let imgs = [];
    if (files.length) {
      imgs = await filesToImgs(Array.from(files));
    }
    imgs = imgs.filter((item) => {
      let checkSizeSucc = item.el.width == IMAGE_W && item.el.height == IMAGE_H;
      if (!checkSizeSucc) {
        message.info(`${item.name} image size unnormal[${IMAGE_W},${IMAGE_H}], neglected`);
      }
      return checkSizeSucc;
    });
    imgs.forEach((item) => {
      item.label = '';
      item.uid = guid();
    });
    imgList4.value.push(...imgs);
    imgInptValue4.value = [];
  };
  const handleFileChange5 = async function (e) {
    let files = e.target.files;
    let imgs = [];
    if (files.length) {
      imgs = await filesToImgs(Array.from(files));
    }
    imgs = imgs.filter((item) => {
      let checkSizeSucc = item.el.width == IMAGE_W && item.el.height == IMAGE_H;
      if (!checkSizeSucc) {
        message.info(`${item.name} image size unnormal[${IMAGE_W},${IMAGE_H}], neglected`);
      }
      return checkSizeSucc;
    });
    imgs.forEach((item) => {
      item.label = '';
      item.uid = guid();
    });
    imgList5.value.push(...imgs);
    imgInptValue5.value = [];
  };
  const handleFileChange6 = async function (e) {
    let files = e.target.files;
    let imgs = [];
    if (files.length) {
      imgs = await filesToImgs(Array.from(files));
    }
    imgs = imgs.filter((item) => {
      let checkSizeSucc = item.el.width == IMAGE_W && item.el.height == IMAGE_H;
      if (!checkSizeSucc) {
        message.info(`${item.name} image size unnormal[${IMAGE_W},${IMAGE_H}], neglected`);
      }
      return checkSizeSucc;
    });
    imgs.forEach((item) => {
      item.label = '';
      item.uid = guid();
    });
    imgList6.value.push(...imgs);
    imgInptValue6.value = [];
  };
  const handleFileChange7 = async function (e) {
    let files = e.target.files;
    let imgs = [];
    if (files.length) {
      imgs = await filesToImgs(Array.from(files));
    }
    imgs = imgs.filter((item) => {
      let checkSizeSucc = item.el.width == IMAGE_W && item.el.height == IMAGE_H;
      if (!checkSizeSucc) {
        message.info(`${item.name} image size unnormal[${IMAGE_W},${IMAGE_H}], neglected`);
      }
      return checkSizeSucc;
    });
    imgs.forEach((item) => {
      item.label = '';
      item.uid = guid();
    });
    imgList7.value.push(...imgs);
    imgInptValue7.value = [];
  };
  const handleFileChange8 = async function (e) {
    let files = e.target.files;
    let imgs = [];
    if (files.length) {
      imgs = await filesToImgs(Array.from(files));
    }
    imgs = imgs.filter((item) => {
      let checkSizeSucc = item.el.width == IMAGE_W && item.el.height == IMAGE_H;
      if (!checkSizeSucc) {
        message.info(`${item.name} image size unnormal[${IMAGE_W},${IMAGE_H}], neglected`);
      }
      return checkSizeSucc;
    });
    imgs.forEach((item) => {
      item.label = '';
      item.uid = guid();
    });
    imgList8.value.push(...imgs);
    imgInptValue8.value = [];
  };
  const handleFileChange9 = async function (e) {
    let files = e.target.files;
    let imgs = [];
    if (files.length) {
      imgs = await filesToImgs(Array.from(files));
    }
    imgs = imgs.filter((item) => {
      let checkSizeSucc = item.el.width == IMAGE_W && item.el.height == IMAGE_H;
      if (!checkSizeSucc) {
        message.info(`${item.name} image size unnormal[${IMAGE_W},${IMAGE_H}], neglected`);
      }
      return checkSizeSucc;
    });
    imgs.forEach((item) => {
      item.label = '';
      item.uid = guid();
    });
    imgList9.value.push(...imgs);
    imgInptValue9.value = [];
  };
  // clear the data
  function clearModalInput() {
    fillAllLabel.value = false;
    fillAllLabelValue.value = '';
    imgList.value = [];
  }
  onMounted(() => {
    testModel.value = useTestModel(preCanvas.value, model);
    // setUpMqtt();
  });
</script>
<style scoped lang="scss">
  .train-box {
    width: 100%;
    height: 100%;
    min-height: 500px;
    position: relative;
    .train-progress-box {
      position: absolute;
      top: 0;
      left: 80px;
      bottom: 0;
      width: 50%;
      overflow: auto;
      background-color: #f2f2f2;
    }
    .oprate-box {
      position: absolute;
      top: 0;
      right: 0;
      bottom: 0;
      width: 50%;
      overflow: auto;
      border-left: 1px dashed #aaa;
      background-color: #f2f2f2;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      .ant-btn {
        margin-bottom: 20px;
      }
    }
  }

  .load-data-box {
    margin-top: 10px;
    .slect-fill-all {
      // display: flex;
      // align-items: center;
      margin: 20px 0;
      padding-bottom: 20px;
      .ant-input {
        margin-top: 10px;
      }
      border-bottom: 2px solid rgb(211,211,211);
    }
    ul {
      max-height: 500px;
      overflow: auto;
    }
    ul,
    li {
      list-style: none;
      padding: 0;
      margin: 0;
      margin-top: 10px;
    }
    li {
      display: flex;
      align-items: center;
      img {
        width: 28px;
        height: 28px;
        margin-right: 20px;
      }
      padding: 10px;
      background-color: #f6f6f6;
      border-radius: 10px;
      label {
        display: flex;
        align-items: center;
      }
    }
  }
</style>
