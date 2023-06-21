# 插件功能定制，参数详解
本插件支持以下参数的配置：这里展示的配置都是默认配置
```json
{
    "showLevel": true,
    "associatedWithSummary": true,
    "printLog": false,
    "multipleH1": true,
    "mode": "float",
    "showGoTop":true,
    "float": {
        "floatIcon": "fa fa-navicon",
        "showLevelIcon": false,
        "level1Icon": "fa fa-hand-o-right",
        "level2Icon": "fa fa-hand-o-right",
        "level3Icon": "fa fa-hand-o-right"
    },
    "pageTop": {
        "showLevelIcon": false,
        "level1Icon": "fa fa-hand-o-right",
        "level2Icon": "fa fa-hand-o-right",
        "level3Icon": "fa fa-hand-o-right"
    }
}
```
## showLevel : TYPE:boolean。
    标题是否显示层级序号.页面标题和导航中的标题都会加上层级显示。（控制页面标题是否被重写）

```
---- xx.md ---
# h1
## h2
### h3

显示层级后的页面效果如下：
1. h1
1.1. h2
1.1.1 h3
```
## associatedWithSummary : TYPE:boolean
    页面内的序号是否与 summary.md 中官方默认主题生成的序号相关联。
```
如果你打开了官网默认主题中的层级显示：
 "pluginsConfig": {
        "anchor-navigation-ex": {
           "associatedWithSummary":true
        },
        "theme-default": {
            "showLevel": true
        }
 }
 那么这样写：

 ----- SUMMARY.md ------
 # Summary

* [安装](chapter/install.md)
* [命令](chapter/command.md)
* [配置](chapter/bookjson.md)
* [插件](chapter/plugin.md)
    * [prismjs 代码高亮](chapter/plugin/prismjs.md)
    * [ace 代码高亮编辑](chapter/plugin/ace.md)
    * [navigator 页面导航](chapter/plugin/navigator.md)

 ----- chapter/redis/cluster.md ------
 # redis集群的准备
 ## zlib
 1. 安装redis-cluster依赖:redis-cluster的依赖库在使用时有兼容问题,在reshard时会遇到各种错误,请按指定版本安装.
 2. 确保系统安装zlib,否则gem install会报(no such file to load -- zlib)

 ...
```
那么最终效果如下：
  ![image](https://raw.githubusercontent.com/zq99299/gitbook-plugin-anchor-navigation-ex/master/doc/images/层级关联显示.png)

## printLog : TYPE:boolean （V1.0.6+）
是否打印处理日志,在排查生成book失败的时候很有用，能知道是哪一个文件出的错
如下图：使用了别的插件，但是只打印了出错的信息，不知道是哪一个文件。开启该选项，就能知道了
![image](https://raw.githubusercontent.com/zq99299/gitbook-plugin-anchor-navigation-ex/master/doc/images/printlog.png)

## multipleH1 : TYPE:boolean  （V1.0.6+）
是否是多h1模式？一般正常的书籍一个章节只有一个h1标签，也就是一个md文件一个标签。如果您的书籍是这种正常模式，请关闭该选项=false
最大的区别如下
```
---- multipleH1=true----
# 多H1标题1
## 子标题
# 多H1标题2
## 子标题
## 子标题

生成的标题如下：
1. 多H1标题1
1.1. 子标题
2. 多H1标题2
2.1. 子标题
2.2. 子标题

---- multipleH1=false----
# 多H1标题1
## 子标题
# 多H1标题2
## 子标题
## 子标题

生成的标题如下：
多H1标题1
1. 子标题
多H1标题2
1. 子标题
2. 子标题

看出区别了么？正常书籍模式，也就是只有一个h1的情况下，这个展示的排序序号更符合我们的需求。
```

## mode
导航模式：分为三种

1. float ：浮动导航
2. pageTop ：页面内部顶部导航
3. '' : 不显示导航

## float
mode = float的时候以下配置生效
```
    float: { //浮动导航设置
        "floatIcon": "fa fa-navicon", // 配置导航图标，如果你喜欢原先的 锚 图标可以配置为 fa-anchor
        "showLevelIcon": false,  //是否显示层级图标
        "level1Icon": "fa fa-hand-o-right", //层级的图标css
        "level2Icon": "fa fa-hand-o-right",
        "level3Icon": "fa fa-hand-o-right"
    }
```
图标使用官网默认主题引入的css `http://fontawesome.dashgame.com/`

## pageTop
mode = pageTop的时候以下配置生效
```
pageTop: {
           showLevelIcon: false,
           level1Icon: "fa fa-hand-o-right",
           level2Icon: "fa fa-hand-o-right",
           level3Icon: "fa fa-hand-o-right"
       }
```
## showGoTop  : TYPE:boolean （V1.0.11+）

把返回顶部按钮独立出来了，为true的时候显示返回顶部按钮

# 额外功能支持

- 在页面中增加`<extoc></extoc>`标签，会在此处生成TOC目录。
- 在页面中增加`<!-- ex_nonav -->`标识，会让此页面不生成悬浮导航

    在首页、介绍页等地方可以使用该功能，能屏蔽一些代码，因为这些地方不能加载css
