(ns ml-model
  (:require [clojure.core.matrix :as matrix :refer [dot transpose exp]]
            [clojure.core.matrix.operators :refer :all]))

;; (matrix/set-current-implementation :vectorz)

(def training-data
  ;; input => output
  [[0 0 1]   [0]
   [0 1 1]   [1]
   [1 0 1]   [1]
   [1 1 1]   [0]
   [0 0 0]   [0]
   [0 1 0]   [1]
   [1 0 0]   [1]
   [1 1 0]   [0]
   [1 1 0]   [1]])

(def training-input
  (take-nth 2 training-data))

(def training-output
  (take-nth 2 (rest training-data)))

(defn matrix-of
  "Return a matrix of results of a function"
  [fn x y]
  (matrix/array (repeatedly x #(repeatedly y fn))))

(defn random-synapse
  "Random float between -1 and 1"
  [] (dec (rand 2)))

;; synapses are mutable so I'm using atoms
(def synapses-0 (atom (matrix-of random-synapse 3 5)))
(def synapses-1 (atom (matrix-of random-synapse 5 1)))

(defn activation
  "Sigmoid function"
  [x] (/ 1 (+ 1 (exp (- x)))))

(defn derivative
  "Derivative of sigmoid"
  [x] (* x (- 1 x)))

(defn layer
  "The layers in our network are a curried function of weights and inputs"
  [weights]
  (fn [inputs]
    (activation (dot inputs weights))))

(defn network [individual]
  (let [s0 (first (:synapses individual))
        s1 (second (:synapses individual))]
    (comp (layer s1) (layer s0))))

;; Thats all we need for forward propagation.
;; We can now use our network as a function which takes training input
;; and returns a predicted output.
(network training-input)


;; The network's error function is simply the difference between the known
;; training outputs and the results of our network fn on training input.
(def errors (fn [] (- training-output (network training-input))))

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

(defn deltas [cost-f synapses]
  (let [s0 (first synapses)
        s1 (second synapses)]
    (reduce
     (fn [deltas layer]
       (conj deltas
             (* (derivative layer)
                (if (empty? deltas)
                  (cost-f)
                  (dot (last deltas) (transpose s1))))))
     [] [(network training-input) ((layer s0) training-input)])))

(defn gradient-descent [individual cost-fn iterations]
  (let [{:keys [synapses learning-rate]} individual]
    (loop [i iterations
           s synapses]
      (if (zero? i)
        (assoc individual :synapses s)
        (let [deltas (deltas cost-fn s)
              s1-updated (+ (first s) (* learning-rate (dot (transpose ((layer (first s)) training-input))
                                                            (first deltas))))
              s0-updated (+ (second s) (* learning-rate (dot (transpose training-input)
                                                             (second deltas))))]
          (recur (dec i) [s0-updated s1-updated]))))))

(defn individual [learning_rate]
  {:synapses [(matrix-of random-synapse 3 5) (matrix-of random-synapse 5 1)]
   :learning-rate learning_rate
  }
  )

(defn new_individual []
  (individual (rand))
  )

(defn mutate [indiv]
  (let [new-rate (+ (/ (rand) 100) (:learning-rate indiv))]
    (assoc indiv :learning-rate new-rate)))

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
  (loop [population (repeatedly population-size
                                #(new_individual))
         generation 0]
    (report generation population)
    (if (>= generation generations)
      (fittest population)
      (recur (conj (repeatedly (dec population-size)
                               #(mutate (tournament_select population)))
                   (fittest population))
             (inc generation)))))

(evolve-learning-rate 10 10)