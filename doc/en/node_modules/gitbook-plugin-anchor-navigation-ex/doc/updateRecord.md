# Update record
## v 1.0.13 - 2019-01-16
- [#38](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/38) 合并该 pr，增加以下功能
- fix bug : 当 md 中 h1,h2,h3 非正常顺序的时候导致 toc 缺失，修复后 toc 展示不会漏掉，可以看出来该标题是不正常的 V1.0.14+
- fix bug : 在上一条 fix bug 中，没有考虑到 showLevel 情况下的序号显示

## v 1.0.12 - 2018-09-17
- [#36](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/36) 合并该 pr，增加以下功能
- config.float.floatIcon 可以配置浮动导航的悬浮图标样式  V1.0.12+
- 在页面中增加`<!-- ex_nolevel -->`不会在该页面生成层级序号 V1.0.12+

## v 1.0.11 - 2018-08-22
- [#33](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/33) 把返回顶部按钮独立出来了，为true的时候显示返回顶部按钮

## v1.0.9 - 2017-08-03
- fix bug: [#26](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/26) 当配置mode:""时，页面标题没有重写

## v1.0.8 - 2017-08-03
- fix bug: [#26](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/26) mode:"" 配置无效，还会在页面顶部生成导航

## v1.0.5 - 2017-07-14
- 新增：printLog 配置选项，如果为true的话，则编译的时候打印当前正在处理的md文件路径，如果处理失败，也好知道是哪一个文件
- 新增：multipleH1 配置选项，如果为true的话，将按照一个md文件有多个H1标题处理，为false的话，则按照一个md文件只包含一个h1标题处理，最大的区别就是去掉了丑陋的1.xxx 中的1.

## v1.0.4 - 2017-06-02
此更新主要是修复标题重复后的策略，和修复上一个版本只处理了h1-h3标题重复的bug [#19](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/19)
- 修改
    - 修改ID生成方式
    - 简化Toc函数，优化代码加快运行速度
- 记过
    - 优先使用标题内容作为ID
    - 保留自定义ID [#18](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/18)
    - 重复ID自动递增后缀 [#6](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/6)

其他代码性能稍微优化
- 使用`<!-- ex_nonav -->`注释让页面不显示导航[#15](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/15)

## v1.0.2 - 2017-05-15
- 在页面中增加`<extoc></extoc>`标签，会在此处生成TOC目录(该目录暂时与mode: "pageTop"模式生成的一致)[#17](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/17)


## v1.0.0 - 2017-03-09
- [#7](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/7)
- [#8](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/8)
- [#9](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/9)
- [#10](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/10)
- [#11](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/11)
- [#12](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/12)
- [#13](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/13)
- [#14](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/14)

## v0.3.2 - 2017-03-08
- 新增配置选项-页面序号是否与官方SUMMARY中生成的序号相关联

## v0.3.0 - 2017-03-06
- 官方层级显示功能 与  每页 相关联显示功能[#4](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/4)


## v0.2.7 - 2017-03-01
- fix bug: 锚链接索引唯一 [#6](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/6)

## v0.2.6 - 2017-03-01
- fix bug: [#5](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/pull/5)

## v0.2.5 - 2017-02-17

1. 进一步优化悬浮导航的样式，和官方默认主题保持一致，更加美观，和格调统一
2. 增加 悬浮导航toc标题前的图标定制 [#2](https://github.com/zq99299/gitbook-plugin-anchor-navigation-ex/issues/2)

## v0.1.9 - 2017-02-17
优化悬浮导航的样式

1. 将阴影缩小，面板背景略微透明
2. 文字显示为 14 px
3. 标题编号 加粗显示
4. 适配 官方默认3套主题样式。导航样式将随着皮肤主题的样式变换而变换

## v0.1.8 - 2017-02-09
* 更换锚点图标显示，更换为css样式。不然 pdf生成的时候会失败

## v0.1.7 - 2017-02-09
* css 命名重构
* 更换锚点图标显示，更换为github一致的svg图标
* 生成的目录增加到页面顶端，在某些情况下，会在页面底部来一个导航。很不美观，如：
  - 在gitbook首页的时候因为不会加载插件的css效果
  - 在生成pdf的时候，css没法被加载

## 2017-02-08
* 重构项目结构

## 2017-02-07
* 在源码中使用了 let 等es6的语法，在本地使用ok，在gitbook上报错：PluginError: Error with plugin "anchor-navigation-ex": Block-scoped declarations (let, const, function, class) not yet supported outside strict mode。不知道是为何，还是改成 var 来声明吧

## 2017-02-06
* 完全重写代码
* 合并锚点和悬浮导航效果，现在只需要引入一个插件了 gitbook-plugin-anchor-navigation-ex

## 2017-01-18
* 页面没有h[1-3] 标签生成失败

## 2017-01-22
* 2017-01-18 提交的有问题。重新修复
