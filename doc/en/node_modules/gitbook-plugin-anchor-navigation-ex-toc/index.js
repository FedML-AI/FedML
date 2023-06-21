var _start = require('./assets/lib/plugin');
var Config = require('./assets/lib/config');
require('./assets/lib/log');
module.exports = {
    book: {
        assets: "./assets",
        css: ["style/plugin.css"]
    },
    hooks: {
        "init": function () {
            Config.handlerAll(this);
        },
        "page": function (page) {
            if (Config.config.printLog) {
                console.info("INFO:".info + "当前正在处理:" + page.path)
            }
            var bookIns = this;
            _start(bookIns, page);
            return page;
        }
    }
};