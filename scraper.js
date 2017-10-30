var gplay = require('google-play-scraper');
var fs = require('fs')
var Promise = require('promise');

var read = Promise.denodeify(fs.readFile)
var write = Promise.denodeify(fs.writeFile)

fs.readFile('appid.txt',function read(err,data) {
	if (err) {
        throw err;
    }
    
    content = data;
    for (x in content) {
    	console.log(x);
    	//jsonwriter(x);
    }
    // Invoke the next step here however you like
    //console.log(content);
       // Put all of the code here (not the best solution)
    //jsonwriter('com.jolansky.glazesimplicity.theme');          // Or put the next step in a function and invoke it

	// body...
});

function jsonwriter(appid){

gplay.app({appId: appid})
  .then(function (str) {
    return write('foo.json', JSON.stringify(str, null, '  '), 'utf8')
  })

}
