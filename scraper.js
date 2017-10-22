var gplay = require('google-play-scraper');
var fs = require('fs')
var Promise = require('promise');

var read = Promise.denodeify(fs.readFile)
var write = Promise.denodeify(fs.writeFile)

gplay.app({appId: 'com.dxco.pandavszombies'})
  .then(function (str) {
    return write('foo.json', JSON.stringify(str, null, '  '), 'utf8')
  })