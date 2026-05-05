(ns mini.playground-no-other-searches
  (:require [clojure.core.matrix :as matrix :refer [dot transpose exp]]
            [clojure.core.matrix.operators :refer :all]
            [clojure.data.csv :as csv]
            [clojure.java.io :as io]))

(defn read-csv [file-path]
  (with-open [reader (io/reader file-path)]
    (doall
     (csv/read-csv reader))))

(println (read-csv "banknote_authentication.csv"))
(def raw-data (vec (shuffle (read-csv "banknote_authentication.csv"))))
 
(def input
  (mapv (fn [row]
          (mapv #(Double/parseDouble %) (take 4 row))) ;; first 4 cols as input
        raw-data))


(def label->one-hot {"0" [1 0]
                     "1" [0 1]})

(def output
  (mapv (fn [row] (label->one-hot (last row))) raw-data))

(def training-input
  (vec (take (int (* 0.8 (count input))) input)))

(def test-input
  (let [n (count input)
        start (int (* 0.8 n))
        end   (int (* 0.9 n))]
    (subvec (vec input) start end)))

(def validation-input
  (let [n (count input)
        start (int (* 0.9 n))]
    (subvec (vec input) start n)))

(def training-output
  (vec (take (int (* 0.8 (count input))) output)))

(def test-output
  (let [n (count output)
        start (int (* 0.8 n))
        end   (int (* 0.9 n))]
    (subvec (vec output) start end)))

(def validation-output
  (let [n (count output)
        start (int (* 0.9 n))]
    (subvec (vec output) start n)))

test-input
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

(defn dropout-mask
  [x rate]
  (let [keep-prob (- 1 rate)]
    (matrix/emap (fn [_]
                   (if (< (rand) keep-prob)
                     (/ 1 keep-prob)
                     0))
                 x)))

(defn apply-dropout [x mask]
  (matrix/mul x mask))

(defn layer
  "The layers in our network are a curried function of weights and inputs"
  [weights activation-fn]
  (fn [inputs]
    (activation-fn (dot inputs weights))))

(defn network [individual]
  (let [s0 (first (:synapses individual))
        s1 (second (:synapses individual))
        activation-fn (:fn (:activation individual))
        dropout-rate (:dropout-rate individual)]
   (fn [inputs]
     (let [l1 (activation-fn (dot inputs s0))
           mask (dropout-mask l1 dropout-rate)
           l1-drop (apply-dropout l1 mask)
           l2 (activation-fn (dot l1-drop s1))]
       l2))))

(defn network-no-dropout [individual]
  (let [s0 (first (:synapses individual))
        s1 (second (:synapses individual))
        activation-fn (:fn (:activation individual))]
    (fn [inputs]
      (let [l1 (activation-fn (dot inputs s0))
            l2 (activation-fn (dot l1 s1))]
        l2))))

;; Thats all we need for forward propagation.
;; We can now use our network as a function which takes training input
;; and returns a predicted output.
(network training-input)

(defn mean-error [numbers]
  (let [absolutes (map #(if (> 0 %) (- %) %) (flatten numbers))]
    (/ (apply + absolutes) (count absolutes))))

(defn maxIndex [v]
  (.indexOf (vec v) (apply max (vec v))))

(defn accuracy
  [output predicted]
  (let [pred-classes (map maxIndex predicted)
        actual-classes (map maxIndex output)]
    (double (/ (count (filter true? (map = pred-classes actual-classes)))
               (count actual-classes)))))

(defn f1_binary
  [output predicted]
  (let [TP (count (filter (fn [[a b]] (and (= a 1) (= b 1))) (map vector output predicted)))
        FP (count (filter (fn [[a b]] (and (= a 0) (= b 1))) (map vector output predicted)))
        FN (count (filter (fn [[a b]] (and (= a 1) (= b 0))) (map vector output predicted)))
        precision (/ TP (max 1 (+ TP FP)))
        recall (/ TP (max 1 (+ TP FN)))]
    (double (/ (* 2 precision recall) (max 1e-15 (+ precision recall))))))

(defn create-binary [size index-list]
  (mapv (fn [idx]
          (assoc (vec (repeat size 0)) idx 1))
        index-list))

(defn f1
  [output predicted]
  (let [max_indices (map maxIndex predicted)
        len (count (first predicted))
        binary_predicted (create-binary len max_indices)
        scores (map f1_binary (apply map vector output)
                    (apply map vector binary_predicted))]
    (/ (apply + scores) (count scores))))

(defn fitness [individual]
  ;; (network individual) returns a function. 
  ;; We then call that function with training-input.
  (let [prediction ((network-no-dropout individual) test-input)]
    (accuracy test-output prediction)))

(defn validation_fitness [individual]
  ;; (network individual) returns a function. 
  ;; We then call that function with training-input.
  (let [prediction ((network-no-dropout individual) validation-input)]
    (accuracy validation-output prediction)))

(defn fittest
  "Returns the fittest of the given individuals. We want higher accuracy
  Change max-key to min-key if you want to use error instead."
  [individuals]
  (apply max-key fitness individuals))

;; The mean-error function just gives a single value to represent how
;; accurate our network's are. This is useful for debuggin but is not used
;; in the training algorithm.


;; To actually train out network we'll use an algorithm called gradient
;; descent.
;; For each layer we need to multiply the derivative of our layer function
;; in relation to the weights by the size of the error.

(def raw_errors_cost (fn [individual] (- training-output ((network individual) training-input))))

(defn cross-entropy-loss [predicted actual]
  (let [epsilon 1e-15  ;; prevent log(0)
        clipped (matrix/emap #(max epsilon (min (- 1 epsilon) %)) predicted)
        log-pred (matrix/emap #(Math/log %) clipped)]
    (- (/ (apply + (flatten (matrix/mul actual log-pred)))
          (count predicted)))))

(defn deltas [cost-f individual batch-input]
  (let [s0 (first (get individual :synapses))
        s1 (second (get individual :synapses))
        activation-fn (:fn (:activation individual))
        derivative (:deriv (:activation individual))
        dropout-rate (:dropout-rate individual)

        l0 batch-input
        l1 (activation-fn (dot l0 s0))
        mask (dropout-mask l1 dropout-rate)
        l1-drop (apply-dropout l1 mask)

        l2 (activation-fn (dot l1-drop s1))

        l2-error (cost-f individual)
        l2-delta (* l2-error (derivative l2))

        l1-error (dot l2-delta (transpose s1))

        l1-delta (* (apply-dropout l1-error mask)
                    (derivative l1))]
    [l2-delta l1-delta]))

(defn gradient-descent [individual iterations]
  (let [{:keys [synapses learning-rate activation momentum batch-size]} individual
        activation-fn (:fn activation)
        n (count training-input)]
    (loop [i iterations
           s synapses
           v [(matrix/mul 0 (first synapses))
              (matrix/mul 0 (second synapses))]]
      (if (zero? i)
        (assoc individual :synapses s)
        (let [indices    (vec (repeatedly batch-size #(rand-int n)))
              batch-in   (mapv training-input indices)
              batch-out  (mapv training-output indices)
              batch-cost (fn [ind] (matrix/sub batch-out ((network ind) batch-in)))
              ind        (assoc individual :synapses s)
              d          (deltas batch-cost ind batch-in)
              s0         (first s)
              s1         (second s)
              v0         (first v)
              v1         (second v)
              g0         (dot (transpose batch-in) (second d))
              g1         (dot (transpose ((layer s0 activation-fn) batch-in)) (first d))
              new-v0     (matrix/add (matrix/mul momentum v0) (matrix/mul learning-rate g0))
              new-v1     (matrix/add (matrix/mul momentum v1) (matrix/mul learning-rate g1))]
          (recur (dec i)
                 [(matrix/add s0 new-v0) (matrix/add s1 new-v1)]
                 [new-v0 new-v1]))))))


(defn individual [learning_rate activation iterations hidden-size init-scale 
                  momentum batch-size dropout-rate]
  (gradient-descent {:synapses [(matrix-of #(* init-scale (dec (rand 2))) 4 hidden-size)
                                (matrix-of #(* init-scale (dec (rand 2))) hidden-size 2)]
                     :learning-rate learning_rate
                     :activation activation
                     :iterations iterations
                     :hidden-size hidden-size
                     :init-scale init-scale
                     :momentum momentum
                     :batch-size batch-size
                     :dropout-rate dropout-rate} iterations))

(defn new_individual []
  (individual (rand) (rand-activation-fn) 5 (+ 2 (rand-int 9)) (+ 0.5 (rand))
              (+ 0.8 (* 0.19 (rand))) (+ 10 (rand-int 41)) (rand)))

;; Probability of differing activation function than parents
(def mutation-rate 0.1)

(defn mutate [indiv]
  (let [new-rate        (max 0.0001 (+ (* (rand-nth [1 -1]) (/ (rand) 100)) (:learning-rate indiv)))
        new-activation  (if (< (rand) mutation-rate)
                          (rand-activation-fn)
                          (:activation indiv))
        new-iterations  (max 1 (+ (rand-nth [1 -1]) (:iterations indiv)))
        new-hidden-size (max 1 (+ (rand-nth [1 -1]) (or (:hidden-size indiv) 5)))
        new-init-scale  (max 0.1 (+ (* (rand-nth [1 -1]) (/ (rand) 10)) (or (:init-scale indiv) 1.0)))
        new-momentum    (min 0.999 (max 0.0 (+ (* (rand-nth [1 -1]) (* 0.05 (rand))) (or (:momentum indiv) 0.9))))
        new-batch-size  (int (max 1 (+ (* (rand-nth [1 -1]) (+ 1 (rand-int 5))) (or (:batch-size indiv) 32))))
        new-dropout (-> (:dropout-rate indiv)
                        (+ (- (rand 0.1) 0.05))
                        (max 0.0)
                        (min 0.9))]
    (individual new-rate new-activation new-iterations new-hidden-size new-init-scale new-momentum new-batch-size new-dropout)))

(defn crossover [indiv1 indiv2]
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
                  (:iterations indiv2))
                (if (= 1 (rand-int 2))
                  (:hidden-size indiv1)
                  (:hidden-size indiv2))
                (if (= 1 (rand-int 2))
                  (:init-scale indiv1)
                  (:init-scale indiv2))
                (if (= 1 (rand-int 2))
                  (:momentum indiv1)
                  (:momentum indiv2))
                (if (= 1 (rand-int 2))
                  (:batch-size indiv1)
                  (:batch-size indiv2))
                (if (= 1 (rand-int 2))
                  (:dropout-rate indiv1)
                  (:dropout-rate indiv2)))))

(defn report
  "Prints a report on the status of the population at the given generation."
  [generation population]
  (let [best (fittest population)]
    (println {:generation generation :fitness (fitness best)
              :valiation_fitness (validation_fitness best)})))

(defn tournament_select
  "Returns an individual selected from population using a tournament."
  [population]
  (fittest (repeatedly 2 #(rand-nth population))))

(defn evolve-learning-rate
  [population-size generations]
  (loop [population (doall (pmap (fn [_] (new_individual)) (range population-size)))
         generation 0]
    (report generation population)
    (if (>= generation generations)
      (fittest population)
      (let [elite       (fittest population)
            n-children  (dec population-size)
            n-mutation  (quot n-children 2)
            n-crossover (- n-children n-mutation)
            mutated     (doall (pmap (fn [_] (mutate (tournament_select population)))
                                     (range n-mutation)))
            crossed     (doall (pmap (fn [_] (crossover (tournament_select population)
                                                        (tournament_select population)))
                                     (range n-crossover)))]
        (recur (conj (concat mutated crossed) elite)
               (inc generation))))))

(defn make-individual [lr act iters hidden-size init-scale momentum batch-size dropout]
  (let [i (individual lr act iters hidden-size init-scale momentum batch-size dropout)]
    (assoc i :fitness (fitness i))))

(defn rand-individual []
  (let [i (new_individual)] (assoc i :fitness (fitness i))))

(defn linspace [lo hi n]
  (if (= n 1) [lo]
    (map #(+ lo (* % (/ (- hi lo) (dec n)))) (range n))))

(defn grid-search
  ([] (grid-search 1000))
  ([budget]
   (let [n      (max 2 (int (Math/pow budget (/ 1.0 7))))
         combos (for [lr  (linspace 0.0001 1.0 n)
                      do  (linspace 0.0001 1.0 n)
                      it  (map int (linspace 1 20 n))
                      hs  (map int (linspace 2 10 n))
                      sc  (linspace 0.5 1.5 n)
                      mo  (linspace 0.8 0.999 n)
                      bs  (map int (linspace 10 50 n))
                      act activation]
                  [lr act it hs sc mo bs do])
         pop    (->> combos
                     (take budget)
                     (pmap (fn [[lr act it hs sc mo bs do]]
                             (make-individual lr act it hs sc mo bs do)))
                     doall)
         best   (apply max-key :fitness pop)]
     (println {:search :grid :evals (count pop) :best-fitness (:fitness best)})
     best)))

(defn random-search
  ([] (random-search 1000))
  ([budget]
   (let [pop  (doall (pmap (fn [_] (rand-individual)) (range budget)))
         best (apply max-key :fitness pop)]
     (println {:search :random :evals budget :best-fitness (:fitness best)})
     best)))

(defn anneal
  ([] (anneal 1000))
  ([budget]
   (let [t0    1.0
         t-min 0.001
         decay (Math/pow (/ t-min t0) (/ 1.0 budget))
         clamp (fn [lo hi x] (max lo (min hi x)))
         perturb (fn [indiv t]
                   (make-individual
                    (clamp 0.0001 1.0 (+ (:learning-rate indiv) (* t (- (rand) 0.5) 2.0)))
                    (if (< (rand) (* t 0.3)) (rand-activation-fn) (:activation indiv))
                    (int (clamp 1 20  (+ (:iterations indiv)  (* t (rand-nth [-1 0 1]) 5))))
                    (int (clamp 2 10  (+ (:hidden-size indiv)  (* t (rand-nth [-1 0 1]) 3))))
                    (clamp 0.1 2.0    (+ (:init-scale indiv)  (* t (- (rand) 0.5))))
                    (clamp 0.0 0.999  (+ (:momentum indiv)    (* t (- (rand) 0.5) 0.4)))
                    (int (clamp 10 50 (+ (:batch-size indiv)  (* t (rand-nth [-1 0 1]) 10))))
                    (clamp 0.0001 1.0 (+ (:dropout-rate indiv) (* t (- (rand) 0.5) 2.0)))))]
     (loop [current (rand-individual)
            best    current
            t       t0
            i       (dec budget)]
       (if (zero? i)
         (do (println {:search :annealing :evals budget :best-fitness (:fitness best)})
             best)
         (let [candidate (perturb current t)
               delta     (- (:fitness current) (:fitness candidate))
               accept?   (or (neg? delta) (< (rand) (Math/exp (- (/ delta t)))))
               next-cur  (if accept? candidate current)
               next-best (if (> (:fitness candidate) (:fitness best)) candidate best)]
           (recur next-cur next-best (* t decay) (dec i))))))))

(defn measure [label f]
  (let [rt         (Runtime/getRuntime)
        _          (System/gc)
        mem-before (- (.totalMemory rt) (.freeMemory rt))
        start      (System/currentTimeMillis)
        result     (f)
        elapsed-ms (- (System/currentTimeMillis) start)
        mem-after  (- (.totalMemory rt) (.freeMemory rt))
        heap-mb    (quot (- mem-after mem-before) (* 1024 1024))]
    (println {:strategy label :ms elapsed-ms :heap-delta-mb heap-mb})
    {:result result :ms elapsed-ms :heap-delta-mb heap-mb}))

(defn compare-searches
  ([] (compare-searches 1000))
  ([budget]
   (let [pop-size (int (Math/sqrt budget))
         evo  (measure :evolutionary #(evolve-learning-rate 100 10))
         grid (measure :grid         #(grid-search budget))
         rnd  (measure :random       #(random-search budget))
         ann  (measure :annealing    #(anneal budget))]
     {:evolutionary evo :grid grid :random rnd :annealing ann})))

(defn run [{:keys [budget] :or {budget 1000}}]
  (compare-searches (int budget)))

(run 100)
