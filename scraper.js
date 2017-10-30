var gplay = require('google-play-scraper');
var fs = require('fs')
var Promise = require('promise');
var myArgs = process.argv.slice(2);
var passed_appid = myArgs[0];
var passed_appcount = myArgs[1];
console.log(passed_appid);
var read = Promise.denodeify(fs.readFile);
var write = Promise.denodeify(fs.writeFile);
var dir = './dataset/' + passed_appcount;
if (!fs.existsSync(dir)){
    fs.mkdirSync(dir);
}

gplay.app({appId: passed_appid})
  .then(function (str) {
    
    return write(dir + '/meta.json', JSON.stringify(str, null, '  '), 'utf8')
  })
.then(function (){process.exit()});