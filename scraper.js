var gplay = require('google-play-scraper');
var fs = require('fs')
var Promise = require('promise');
var myArgs = process.argv.slice(2);
var passed_appid = myArgs[0];
var read = Promise.denodeify(fs.readFile)
var write = Promise.denodeify(fs.writeFile)

gplay.app({appId: passed_appid})
  .then(function (str) {
    console.log(str);
    return write(passed_appid + '.json', JSON.stringify(str, null, '  '), 'utf8');
  })
process.exit();