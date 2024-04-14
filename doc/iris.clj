(ns iris
  (:require
   [conceptron.ac :as ac]
   [tech.v3.datatype.functional :as f]
   [tech.v3.dataset :as ds]))

(def iris (ds/->dataset "./doc/iris/iris.data" {:file-type :csv :header-row? false}))

;; | column-0 | column-1 | column-2 | column-3 |       column-4 |
;; |---------:|---------:|---------:|---------:|----------------|
;; |      5.1 |      3.5 |      1.4 |      0.2 |    Iris-setosa |
;; |      4.9 |      3.0 |      1.4 |      0.2 |    Iris-setosa |
;; |      4.7 |      3.2 |      1.3 |      0.2 |    Iris-setosa |
;; |      4.6 |      3.1 |      1.5 |      0.2 |    Iris-setosa |
;; |      5.0 |      3.6 |      1.4 |      0.2 |    Iris-setosa |
;; |      5.4 |      3.9 |      1.7 |      0.4 |    Iris-setosa |
;; |      4.6 |      3.4 |      1.4 |      0.3 |    Iris-setosa |
;; |      5.0 |      3.4 |      1.5 |      0.2 |    Iris-setosa |
;; |      4.4 |      2.9 |      1.4 |      0.2 |    Iris-setosa |
;; |      4.9 |      3.1 |      1.5 |      0.1 |    Iris-setosa |
;; |      ... |      ... |      ... |      ... |            ... |
;; |      6.9 |      3.1 |      5.4 |      2.1 | Iris-virginica |
;; |      6.7 |      3.1 |      5.6 |      2.4 | Iris-virginica |
;; |      6.9 |      3.1 |      5.1 |      2.3 | Iris-virginica |
;; |      5.8 |      2.7 |      5.1 |      1.9 | Iris-virginica |
;; |      6.8 |      3.2 |      5.9 |      2.3 | Iris-virginica |
;; |      6.7 |      3.3 |      5.7 |      2.5 | Iris-virginica |
;; |      6.7 |      3.0 |      5.2 |      2.3 | Iris-virginica |
;; |      6.3 |      2.5 |      5.0 |      1.9 | Iris-virginica |
;; |      6.5 |      3.0 |      5.2 |      2.0 | Iris-virginica |
;; |      6.2 |      3.4 |      5.4 |      2.3 | Iris-virginica |
;; |      5.9 |      3.0 |      5.1 |      1.8 | Iris-virginica |

;; 7. Attribute Information:
;;    1. sepal length in cm
;;    2. sepal width in cm
;;    3. petal length in cm
;;    4. petal width in cm
;;    5. class:
;;       -- Iris Setosa
;;       -- Iris Versicolour
;;       -- Iris Virginica

;; --------------------------------
;; some sensor projection

(keys iris)

(for
    [c ["column-0" "column-1" "column-2" "column-3"]]
  (f/reduce-max (iris c)))

'(7.9 4.4 6.9 2.5)

;;
;; normalize to between 0 and 1
;; make a projection
;; relay nuclues
;;
