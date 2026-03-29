(ns mini.playground)



;; (defn set [length]
;;   (vec (replicate length 0))
;; (print "Set:" set 2)
;; 
;; (defn powerset [set]
;;   for [i (range (count set))] (inc i)
;;   (println (get set i)))
;; 
;; (println powerset ())
;;   

(defn rand-help [m j]
  (vec (repeatedly m #(rand-int j))))

(defn matrixn [rows cols max-val]
  (vec (take rows (repeatedly #(rand-help cols max-val)))))

(matrixn 2 2 10)

;; (def n 9)
;; (def universe (set (range 1 (inc n))))
;; universe
;; (defn powerset
;;   [s]
;;   (if (empty? s)
;;     #{#{}}
;;     (let [element (first s)
;;           sub-powerset (powerset (rest s))]
;;       (union sub-powerset
;;              (set (map #(conj % element) sub-powerset))))))

;; (defn cartesian-product
;;  ([] '(()))
;;  ([& seqs]
;;  (for [prod (apply cartesian-product (rest seqs))
;;        x (first seqs)]
;;    (cons x prod))))
;; 
;; (cartesian-product [1] [1 2] [3 4])
;; (map #(apply + %) (cartesian-product [1] [2]))
;; 
;; (defmacro <|
;;  [f x y]
;;  `(map #(apply ~f %) (cartesian-product ~x ~y)))
;; 
;; (macroexpand
;; '(<| + [1] [2]))
;; 
;; (defn |>
;;  [f x y]
;;  (map #(apply f %) (cartesian-product x y)))
;; 
;; (<| + [1 2] [2 3])
;; (|> + [1 2] [2 3])
;; 
;; (def x [1 2 3])
;; (def y (|> + x [2]))
;; y
;; 
;; (def a [4 7 9])
;; (def b (<| + a [3]))
;; b

; This project has custom configuration.
; See .vscode/settings.json

; If you are new to Calva, you may want to use the command:
; Calva: Create a “Getting Started” REPL project
; which creates a project with a an interactive Calva (and Clojure) guide.