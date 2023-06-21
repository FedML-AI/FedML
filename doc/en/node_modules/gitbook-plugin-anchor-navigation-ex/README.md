# gitbook-plugin-anchor-navigation-ex

[![GitHub issues](https://img.shields.io/github/issues/zq99299/gitbook-plugin-anchor-navigation-ex.svg)](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/issues) [![GitHub issues](https://img.shields.io/github/issues-closed/zq99299/gitbook-plugin-anchor-navigation-ex.svg)](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/issues?q=is%3Aissue+is%3Aclosed) [![GitHub license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://raw.githubusercontent.com/zq99299/gitbook-plugin-anchor-navigation-ex/master/LICENSE) [![npm](https://img.shields.io/npm/v/gitbook-plugin-anchor-navigation-ex.svg)](https://www.npmjs.com/package/gitbook-plugin-anchor-navigation-ex) [![npm](https://img.shields.io/npm/dt/gitbook-plugin-anchor-navigation-ex.svg)](https://www.npmjs.com/package/gitbook-plugin-anchor-navigation-ex)



===============	【DOC：中文，[English](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/blob/master/doc/README_EN.md) 】================

===============	【UpdateRecord：[中文](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/blob/master/doc/updateRecord.md)，[English](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/blob/master/doc/updateRecord-en.md)】================

-----

:exclamation: 该项目停更，转向使用 vuepress 构建笔记了

-----

:exclamation: 插件配置不兼容V1.x 以下版本

# 插件功能
- [x] 给页面H1-H6标题增加锚点效果
- [x] 浮动导航模式
- [x] 页面内顶部导航模式
- [x] 导航标题前的层级图标是否显示，自定义H1-H3的层级图标
- [x] plugins["theme-default"],页面标题层级与官方默认主题的`showLevel`层级关联
- [x] plugins["theme-default"],插件样式支持官网默认主题的三种样式：White、Sepia、Night
- [x] 在页面中增加`<extoc></extoc>`标签，会在此处生成TOC目录
- [x] 在页面中增加`<!-- ex_nonav -->`标签，不会在该页面生成悬浮导航
- [x] config.printLog=true,打印当前的处理进度，排错很有用
- [x] config.multipleH1=false,去掉丑陋的多余的1. 序号（如过您的书籍遵循一个MD文件只有一个H1标签的话）
- [x] config.showGoTop=true,显示返回顶部按钮 V1.0.11+
- [x] config.float.floatIcon 可以配置浮动导航的悬浮图标样式  V1.0.12+
- [x] 在页面中增加`<!-- ex_nolevel -->`不会在该页面生成层级序号 V1.0.12+

# 插件效果
* 风格：极简
* [点击查看效果图](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/blob/master/doc/effectScreenshot.md)


# 怎么使用插件？

在你的 `book.json` 中增加插件：

```
{
  "plugins": [
       "anchor-navigation-ex"
  ]
}
```
然后安装插件:

```
$ gitbook install ./
```

就可以使用了，插件详细配置，[请点击这里查看](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/blob/master/doc/config.md)


或则单独安装插件

```
$ npm install gitbook-plugin-anchor-navigation-ex --save
```

>open npm : https://www.npmjs.com/package/gitbook-plugin-anchor-navigation-ex


# 致敬
本插件集合以下插件的功能，并重写。

1. https://github.com/zhangzq/gitbook-plugin-navigator
2. https://github.com/yaneryou/gitbook-plugin-anchor-navigation

