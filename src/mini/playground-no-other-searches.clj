(ns ml-model
  (:require [clojure.core.matrix :as matrix :refer [dot transpose exp]]
            [clojure.core.matrix.operators :refer :all]
            [clojure.data.csv :as csv]
            [clojure.java.io :as io]))

(defn read-csv [file-path]
  (with-open [reader (io/reader file-path)]
    (doall
     (csv/read-csv reader))))

(println (rest (read-csv "iris.csv")))
(def raw-data (rest (read-csv "iris.csv")))
 
(def training-input
  (mapv (fn [row]
          (mapv #(Double/parseDouble %) (take 4 row))) ;; first 4 cols as input
        raw-data))

(def label->num {"setosa" 0 "versicolor" 1 "virginica" 2})

(def training-output
  (mapv (fn [row]
          [(label->num (last row))])
        raw-data))

(defn matrix-of
  "Return a matrix of results of a function"
  [fn x y]
  (matrix/array (repeatedly x #(repeatedly y fn))))

(defn random-synapse
  "Random float between -1 and 1"
  [] (dec (rand 2)))
;; 
;; (defn activation
;;   "Sigmoid function"
;;   [x] (/ 1 (+ 1 (exp (- x)))))

(defn sigmoid [x]
  (matrix/emap #(/ 1 (+ 1 (exp (- %)))) x))

(defn sigmoid-derivative [x]
  (matrix/emap #(* % (- 1 %)) x))

(defn relu [x]
  (matrix/emap #(max 0 %) x))

;; 1 if x > 0, else 0
(defn relu-derivative [x]
  (matrix/emap #(if (> % 0) 1 0) x))

(defn tanh [x]
  (matrix/emap #(Math/tanh %) x))

;; 1/cos^2(x)
(defn tanh-derivative [x]
  (matrix/emap #(Math/pow (/ 1 (Math/cos %)) 2) x))

(def activation
  [{:fn sigmoid :deriv sigmoid-derivative :name :sigmoid}
   {:fn relu    :deriv relu-derivative    :name :relu}
   {:fn tanh    :deriv tanh-derivative    :name :tanh}])

(defn rand-activation-fn []
  (rand-nth activation))
;; (rand-activation-fn)

(defn layer
  "The layers in our network are a curried function of weights and inputs"
  [weights activation-fn]
  (fn [inputs]
    (activation-fn (dot inputs weights))))

(defn network [individual]
  (let [s0 (first (:synapses individual))
        s1 (second (:synapses individual))
        activation-fn (:fn (:activation individual))]
    (comp (layer s1 activation-fn) (layer s0 activation-fn))))

;; Thats all we need for forward propagation.
;; We can now use our network as a function which takes training input
;; and returns a predicted output.
(network training-input)

(defn mean-error [numbers]
  (let [absolutes (map #(if (> 0 %) (- %) %) (flatten numbers))]
    (/ (apply + absolutes) (count absolutes))))

(defn fitness [individual]
  ;; (network individual) returns a function. 
  ;; We then call that function with training-input.
  (let [prediction ((network individual) training-input)]
    (mean-error (matrix/sub training-output prediction))))

(defn fittest
  "Returns the fittest of the given individuals. We want lower error"
  [individuals]
  (reduce (fn [i1 i2]
            (if (< (fitness i1) (fitness i2))
              i1
              i2))
          individuals))

;; The mean-error function just gives a single value to represent how
;; accurate our network's are. This is useful for debuggin but is not used
;; in the training algorithm.


;; To actually train out network we'll use an algorithm called gradient
;; descent.
;; For each layer we need to multiply the derivative of our layer function
;; in relation to the weights by the size of the error.

(def errors (fn [individual] (- training-output ((network individual) training-input))))

(defn deltas [cost-f individual]
  (let [s0 (first (get individual :synapses))
        s1 (second (get individual :synapses))
        activation-fn (:fn (:activation individual))
        derivative (:deriv (:activation individual))
        l0 training-input
        l1 ((layer s0 activation-fn) l0)
        l2 ((layer s1 activation-fn) l1)]

    (reduce
      (fn [deltas layer]
        (conj deltas
              (* (derivative layer)
                 (if (empty? deltas)
                   (cost-f individual)
                   (dot (last deltas) (transpose s1))))))
      [] [((network individual) training-input) ((layer s0 activation-fn) training-input)])))

(defn gradient-descent [individual cost-fn iterations]
  (let [{:keys [synapses learning-rate activation]} individual
        activation-fn (:fn activation)]
    (loop [i iterations
           s synapses]
      (if (zero? i)
        (assoc individual :synapses s)
        (let [ind      (assoc individual :synapses s)
              d        (deltas cost-fn ind)
              s0       (first s)
              s1       (second s)
              s0-updated (matrix/add s0 (matrix/mul learning-rate
                                                    (dot (transpose training-input)
                                                         (second d))))
              s1-updated (matrix/add s1 (matrix/mul learning-rate
                                                    (dot (transpose ((layer s0 activation-fn) training-input))
                                                         (first d))))]
          (recur (dec i) [s0-updated s1-updated]))))))


(defn individual [learning_rate activation iterations]
  (gradient-descent {:synapses [(matrix-of random-synapse 4 5) (matrix-of random-synapse 5 1)]
   :learning-rate learning_rate
   :activation activation
   :iterations iterations} errors iterations))

(defn new_individual []
  (individual (rand) (rand-activation-fn) 5))

;; Probability of differing activation function than parents
(def mutation-rate 0.1)

(defn mutate [indiv]
  (let [new-rate (+ (/ (rand) 100) (:learning-rate indiv))
        new-activation (if (< (rand) mutation-rate)
                         (rand-activation-fn)
                         (:activation indiv))
        new-iterations (max 1 (+ (rand-nth [1 -1]) (:iterations indiv)))]
    (individual new-rate new-activation new-iterations)))

(defn crossover [indiv1 indiv2]
  ; add more hyperparameters if needed
  (let [activation (or (if (= 1 (rand-int 2))
                         (:activation indiv1)
                         (:activation indiv2))
                       (rand-activation-fn))]
    (individual (if (= 1 (rand-int 2))
                  (:learning-rate indiv1)
                  (:learning-rate indiv2))
                activation
                (if (= 1 (rand-int 2))
                  (:iterations indiv1)
                  (:iterations indiv2)))))

(defn report
  "Prints a report on the status of the population at the given generation."
  [generation population]
  (println {:generation generation :best (fittest population) :fitness (fitness (fittest population))}))

(defn tournament_select
  "Returns an individual selected from population using a tournament."
  [population]
  (fittest (repeatedly 2 #(rand-nth population))))

(defn evolve-learning-rate
  [population-size generations]
  (loop [population (repeatedly population-size #(new_individual))
         generation 0]
    (report generation population)
    (if (>= generation generations)
      (fittest population)
      (let [elite          (fittest population)
            n-children     (dec population-size)
            n-mutation     (quot n-children 2)
            n-crossover    (- n-children n-mutation)
            mutated        (repeatedly n-mutation
                                       #(mutate (tournament_select population)))
            crossed        (repeatedly n-crossover
                                       #(crossover (tournament_select population)
                                                   (tournament_select population)))]
        (recur (conj (concat mutated crossed) elite)
               (inc generation))))))


(evolve-learning-rate 10 10)

;; Accuracy Validation Check

;;  Grid Search
