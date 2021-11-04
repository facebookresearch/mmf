// C implementation of the PHOC respresentation. Converts a string into a PHOC
// feature vector from
// https://github.com/lluisgomez/single-shot-str/blob/master/cphoc/cphoc.c

#include <Python.h>
#include <stdio.h>
#include <stdlib.h>

#define min(X, Y) (((X) < (Y)) ? (X) : (Y))
#define max(X, Y) (((X) > (Y)) ? (X) : (Y))

static PyObject* build_phoc(PyObject* self, PyObject* args) {
  char* word = NULL;
  int ok;
  ok = PyArg_ParseTuple(args, "s", &word);
  if (!ok) {
    return PyErr_Format(
        PyExc_RuntimeError,
        "Failed to parse arguments in build_phoc. Call build_phoc with a single str parameter.");
  }

  float phoc[604] = {.0};

  int index, level, region, i, k, l;

  char* unigrams[36] = {"a", "b", "c", "d", "e", "f", "g", "h", "i",
                        "j", "k", "l", "m", "n", "o", "p", "q", "r",
                        "s", "t", "u", "v", "w", "x", "y", "z", "0",
                        "1", "2", "3", "4", "5", "6", "7", "8", "9"};
  char* bigrams[50] = {"th", "he", "in", "er", "an", "re", "es", "on", "st",
                       "nt", "en", "at", "ed", "nd", "to", "or", "ea", "ti",
                       "ar", "te", "ng", "al", "it", "as", "is", "ha", "et",
                       "se", "ou", "of", "le", "sa", "ve", "ro", "ra", "ri",
                       "hi", "ne", "me", "de", "co", "ta", "ec", "si", "ll",
                       "so", "na", "li", "la", "el"};

  int n = strlen(word);
  for (index = 0; index < n; index++) {
    float char_occ0 = (float)index / (float)n;
    float char_occ1 = (float)(index + 1) / (float)n;
    int char_index = -1;
    for (k = 0; k < 36; k++) {
      if (memcmp(unigrams[k], word + index, 1) == 0) {
        char_index = k;
        break;
      }
    }
    if (char_index == -1) {
      char error_msg[50];
      sprintf(error_msg, "Error: unigram %c is unknown", *(word + index));
      return PyErr_Format(PyExc_RuntimeError, error_msg);
    }
    // check unigram levels
    for (level = 2; level < 6; level++) {
      for (region = 0; region < level; region++) {
        float region_occ0 = (float)region / level;
        float region_occ1 = (float)(region + 1) / level;
        float overlap0 = max(char_occ0, region_occ0);
        float overlap1 = min(char_occ1, region_occ1);
        float kkk = ((overlap1 - overlap0) / (char_occ1 - char_occ0));
        if (kkk >= (float)0.5) {
          int sum = 0;
          for (l = 2; l < 6; l++)
            if (l < level)
              sum += l;
          int feat_vec_index = sum * 36 + region * 36 + char_index;
          phoc[feat_vec_index] = 1;
        }
      }
    }
  }

  // add bigrams
  int ngram_offset = 36 * 14;
  for (i = 0; i < (n - 1); i++) {
    int ngram_index = -1;
    for (k = 0; k < 50; k++) {
      if (memcmp(bigrams[k], word + i, 2) == 0) {
        ngram_index = k;
        break;
      }
    }
    if (ngram_index == -1) {
      continue;
    }
    float ngram_occ0 = (float)i / n;
    float ngram_occ1 = (float)(i + 2) / n;
    level = 2;
    for (region = 0; region < level; region++) {
      float region_occ0 = (float)region / level;
      float region_occ1 = (float)(region + 1) / level;
      float overlap0 = max(ngram_occ0, region_occ0);
      float overlap1 = min(ngram_occ1, region_occ1);
      if ((overlap1 - overlap0) / (ngram_occ1 - ngram_occ0) >= 0.5) {
        phoc[ngram_offset + region * 50 + ngram_index] = 1;
      }
    }
  }

  PyObject* dlist = PyList_New(604);

  for (i = 0; i < 604; i++)
    PyList_SetItem(dlist, i, PyFloat_FromDouble((double)phoc[i]));

  return dlist;
}

static PyObject* getList(PyObject* self, PyObject* args) {
  PyObject* dlist = PyList_New(2);
  PyList_SetItem(dlist, 0, PyFloat_FromDouble(0.00001));
  PyList_SetItem(dlist, 1, PyFloat_FromDouble(42.0));

  return dlist;
}

// Our Module's Function Definition struct
// We require this `NULL` to signal the end of our method
// definition
static PyMethodDef myMethods[] = {
    {"build_phoc", build_phoc, METH_VARARGS, ""},
    {"getList", getList, METH_NOARGS, ""},
    {NULL, NULL, 0, NULL}};

// Our Module Definition struct
static struct PyModuleDef cphoc =
    {PyModuleDef_HEAD_INIT, "cphoc", "cphoc Module", -1, myMethods};

// Initializes our module using our above struct
PyMODINIT_FUNC PyInit_cphoc(void) {
  return PyModule_Create(&cphoc);
}
