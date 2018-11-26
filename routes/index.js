var express = require('express');
var router = express.Router();
let {PythonShell} = require('python-shell')
/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('index', { title: 'Express' });
});



router.get('/show', function(req, res, next) {
  var resultsString;
  let options = {
    mode: 'text',
    pythonOptions: ['-u'], // get print results in real-time
    scriptPath: './',
    args: ['training_data_input_file.txt', '--dropout', '--use-stopwords']
  };

  PythonShell.run('main.py', options, function (err, results) {
    if (err) throw err;

   resultsString = results.toString();
   console.log(resultsString);
   res.send(resultsString);
   res.end();
  });
  res.render('show', function(err, resultsString) {

});
});


module.exports = router;
