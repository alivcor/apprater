var gplay = require('google-play-scraper');
var fs = require('fs')
var Promise = require('promise');
var myArgs = process.argv.slice(2);
var passed_search_term = myArgs[0];
var read = Promise.denodeify(fs.readFile);
var write = Promise.denodeify(fs.writeFile);

console.log(passed_search_term);

var dir = './rawdata';


gplay.search({
        term: passed_search_term,
        num: 250
    }).then(function (str) {
    if(JSON.stringify(str, null, '  ').indexOf("title") > -1) {
        if (!fs.existsSync(dir)){
            fs.mkdirSync(dir);
        }
        return write(dir + '/' + passed_search_term + '.json', JSON.stringify(str, null, '  '), 'utf8')
    } else {
        console.log('Bad response returned !');
        return false
    }

  })
.then(function (){process.exit()});

