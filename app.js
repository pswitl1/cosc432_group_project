var createError = require('http-errors');
var express = require('express');
var path = require('path');
var cookieParser = require('cookie-parser');
var logger = require('morgan');
let {PythonShell} = require('python-shell')
var indexRouter = require('./routes/index');
var usersRouter = require('./routes/users');

var app = express();

/*
let options = {
  mode: 'text',
  pythonOptions: ['-u'], // get print results in real-time
  scriptPath: './',
  args: ['training_data_input_file.txt']
};

PythonShell.run('main.py', options, function (err, results) {
  console.log("running");
  if (err) throw err;
  // results is an array consisting of messages collected during execution
  console.log('results: %j', results);
});
*/
// view engine setup
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'ejs');

app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));

app.use('/', indexRouter);
app.use('/users', usersRouter);

// catch 404 and forward to error handler
app.use(function(req, res, next) {
  next(createError(404));
});

// error handler
app.use(function(err, req, res, next) {
  // set locals, only providing error in development
  res.locals.message = err.message;
  res.locals.error = req.app.get('env') === 'development' ? err : {};

  // render the error page
  res.status(err.status || 500);
  res.render('error');
});

module.exports = app;
