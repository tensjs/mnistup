'use strict';

const fs = require('fs');
const es = require('event-stream');
const zlib = require('zlib');
const path = require('path');
const https = require('https');
const progress = require('progress');
const numjs = require('numjs');

const SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/';
const TRAIN_IMAGES = 'train-images-idx3-ubyte';
const TRAIN_LABELS = 'train-labels-idx1-ubyte';
const TEST_IMAGES = 't10k-images-idx3-ubyte';
const TEST_LABELS = 't10k-labels-idx1-ubyte';

function _download(filename, callback) {
  const local = path.join(__dirname, './data', filename);
  const exists = fs.existsSync(local);
  if (!exists) {
    const sourceUrl = SOURCE_URL + filename + '.gz';
    const dest = fs.createWriteStream(local);
    https.get(sourceUrl).on('response', (res) => {
      const bufs = [];
      const len = parseInt(res.headers['content-length'], 10);
      const bar = new progress(`${filename}: [:bar] :rate/bps :percent :etas`, {
        complete: '=',
        incomplete: ' ',
        width: 20,
        total: len,
      });
      const barCount = es.map((chunk, callback) => {
        bar.tick(chunk.length);
        callback(null, chunk);
      });
      const chunkCollector = es.map((chunk, callback) => {
        bufs.push(chunk);
        callback(null, chunk);
      });
      res.pipe(barCount)
        .pipe(zlib.createUnzip())
        .pipe(dest)
        .on('finish', () => {
          callback(null, Buffer.concat(bufs));
        });
    });
  } else {
    fs.readFile(local, callback);
  }
}

function download(filename) {
  return new Promise((resolve, reject) => {
    _download(filename, (err, data) => {
      err ? reject(err) : resolve(data);
    });
  });
}

function _extract(buffer, block) {
  let offset = 0;
  function _read32() {
    const r = buffer.readUInt32BE(offset);
    offset += 4;
    return r;
  }
  function _read(len) {
    const ab = buffer.buffer.slice(offset, offset + len);
    const r = Buffer.from(ab);
    offset += len;
    return r;
  }

  return block(_read32, _read);
}

function extractImages(buffer) {
  return _extract(buffer, (read32, read) => {
    if (read32() !== 2051)
      throw new Error('Invalid magic number in image file');

    const count = read32();
    const rows = read32();
    const cols = read32();
    const data = read(count * rows * cols);
    console.log(
      `images: count=${count} shape=${rows}x${cols} length=${data.byteLength}`
    );
    return data;
  });
}

function extractLabels(buffer) {
  return _extract(buffer, (read32, read) => {
    if (read32() !== 2049)
      throw new Error('Invalid magic number in label file');

    const count = read32();
    const data = read(count);
    console.log(
      `labels: count=${count} length=${data.byteLength}`
    );
    return data;
  });
}

exports.load = function() {
  return Promise.all([
    download(TRAIN_IMAGES),
    download(TRAIN_LABELS),
    download(TEST_IMAGES),
    download(TEST_LABELS)
  ]).then((buffers) => {
    return buffers.map((data, index) => {
      return index % 2 === 0 ?
        extractImages(data) : extractLabels(data);
    });
  });
};
