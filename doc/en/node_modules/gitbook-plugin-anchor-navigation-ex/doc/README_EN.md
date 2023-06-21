# gitbook-plugin-anchor-navigation-ex

[![GitHub issues](https://img.shields.io/github/issues/zq99299/gitbook-plugin-anchor-navigation-ex.svg)](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/issues) [![GitHub issues](https://img.shields.io/github/issues-closed/zq99299/gitbook-plugin-anchor-navigation-ex.svg)](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/issues?q=is%3Aissue+is%3Aclosed) [![GitHub license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://raw.githubusercontent.com/zq99299/gitbook-plugin-anchor-navigation-ex/master/LICENSE) [![npm](https://img.shields.io/npm/v/gitbook-plugin-anchor-navigation-ex.svg)](https://www.npmjs.com/package/gitbook-plugin-anchor-navigation-ex) [![npm](https://img.shields.io/npm/dt/gitbook-plugin-anchor-navigation-ex.svg)](https://www.npmjs.com/package/gitbook-plugin-anchor-navigation-ex)



===============	【DOC：[中文](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/blob/master/doc/README.md)，English 】================

===============	【UpdateRecord：[中文](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/blob/master/doc/updateRecord.md)，[English](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/blob/master/doc/updateRecord-en.md)】================



-----

:exclamation: The plug-in configuration incompatible V1. X version below

:kissing_heart: My English is not good, this document is through the machine translation, if you don't understand, please read the document in Chinese, the translation itself

# plug-in function
- [x] H1 - H6  to page title increase anchor effect
- [x] floating navigation mode
- [x] page at the top of the navigation mode
- [x] navigation title before the hierarchy of the icon is displayed, the custom of H1, H3 level icon
- [x] plugins["theme-default"],The page title level with the default theme official ` showLevel ` hierarchy correlation
- [x] plugins["theme-default"],Plug-in style website three kinds of style of the default theme：White、Sepia、Night
- [x] Adding `<extoc></extoc>` tags to a page generates the TOC directory here
- [x] Add the `<! - ex_nonav ->` tag to the page without generating a floating navigation on that page
- [x] config.printLog=true, the current progress of the printing process, debugging is very useful
- [x] config.multipleH1=false, remove ugly redundant 1. serial numbers (such as after your books follow a MD file with only one H1 tag)
- [x] config.showGoTop=true,Displays the back top button V1.0.11+
- [x] config.float.floatIcon ，You can configure the float navigation float icon style  V1.0.12+
- [x]  Add the `<!-- ex_nolevel -->` tag to the page，No hierarchical Numbers are generated on this page V1.0.12+

# plug-in effect
* style: a minimalist
* [Click to view rendering](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/blob/master/doc/effectScreenshot.md)


# how to use the plugin?
In your ` book. Json ` add plugins:

```
{
  "plugins": [
       "anchor-navigation-ex"
  ]
}
```
Then install the plugin:

```
$ gitbook install ./
```

Can use the plug-in configuration in detail, [please click here to view](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/blob/master/doc/config-en.md)


Or a separate plugin is installed

```
$ npm install gitbook-plugin-anchor-navigation-ex --save
```

>open npm : https://www.npmjs.com/package/gitbook-plugin-anchor-navigation-ex


# salute
This sets the following plugin functions, and rewrite.

1. https://github.com/zhangzq/gitbook-plugin-navigator
2. https://github.com/yaneryou/gitbook-plugin-anchor-navigation

