;; assembly calculus
(ns conceptron.ac
  (:require
   [clojure.string :as str]
   [tech.v3.datatype :as dtype]
   [tech.v3.datatype.argops :as argpos]
   [tech.v3.tensor :as dtt]
   [tech.v3.datatype.argops :as argops]
   [tech.v3.datatype.functional :as dtype-fn]
   [clojure.set :as set]
   [clojure.java.io :as io]
   [tech.v3.datatype.unary-pred :as unary-pred]
   [tech.v3.datatype.bitmap :as bitmap]))

(require '[fastmath.random :as fm.rand])

(defn outer-product
  ;; Author: Chris Nuernberger
  ;; https://github.com/scicloj/scicloj-data-science-handbook/pull/2#discussion_r548033202
  [f a b]
  (let [a-shape (dtype/shape a)
        b-shape (dtype/shape b)
        a-rdr   (dtype/->reader a)
        b-rdr   (dtype/->reader b)
        n-b     (.lsize b-rdr)
        n-elems (* (.lsize a-rdr) n-b)]
    ;;Doing the cartesian join is easier in linear space
    (-> (dtype/emap
         (fn [^long idx]
           (let [a-idx (quot idx n-b)
                 b-idx (rem idx n-b)]
             (f (a-rdr a-idx) (b-rdr b-idx))))
         :object
         (range n-elems))
        (dtt/reshape (concat a-shape b-shape)))))

(defn eye [n]
  (letfn [(= [a b] (if (clojure.core/= a b) 2 0))]
    (outer-product = (dtt/->tensor (range n))
                   (dtt/->tensor (range n)))))

(defn mask [f t]
  (-> (dtype/emap #(if (f %) 2 0) :int32 t)
      dtt/->tensor
      dtype/->reader))

(defn indices->bool-reader
  [indices max-n]
  (dtype/clone
   (dtype/set-value!
    (dtype/make-container :boolean max-n)
    [indices]
    (dtt/compute-tensor [(count indices)] (constantly 1) :boolean))))

;; ====================== assembly calculus ======================

(defn ->edges
  [n-neurons density]
  (let [m (dtt/compute-tensor [n-neurons n-neurons]
                              (fn [_ _]
                                (< (fm.rand/frand) density))
                              :boolean)
        ;; consider literature on 'autapses'
        ;; perhaps allowing self connections is totally fine.
        diag (dtype-fn/not (eye n-neurons))]
    (dtt/clone (dtype-fn/and m diag))))

(defn ->random-directed-graph
  [n-neurons density]
  (dtt/->tensor (dtype/emap #(if % 2 0) :float32 (->edges n-neurons density))))

(comment
  (into [] (repeatedly 10 #(->edges 1000 0.1))))

(defn synaptic-input
  [weights activations]
  (dtt/reduce-axis
   (dtt/select weights activations)
   dtype-fn/sum
   0
   :float32))

(defn normalize
  [weights]
  ;;         for w, inp in zip(self.input_weights,
  ;;         self.inputs):
  ;;             w /= w.sum(axis=0, keepdims=True)
  (let [normalize-1d
        (fn [d]
          (let [sum (dtype-fn/sum d)]
            (if (zero? sum) d (dtype-fn// d sum))))]
    (dtype/clone (dtt/map-axis weights normalize-1d 0))))

(defn ->neurons [n] (range n))


;; ---------------------
;; Hebbian plasticity dictates that w-ij be increased by a factor of 2 + β at time t + 1
;; if j fires at time t and i fires at time t + 1,
;; --------------

(defn hebbian-plasticity-1
  [{:keys [plasticity weights current-activations
           next-activations]}]
  ;; for w, inp in zip(self.input_weights,
  ;; self.inputs):
  ;;   w[np.ix_(inp, new_activations)] *= 2 +
  ;;   self.plasticity
  (let [w (dtype-fn/* (dtt/select weights
                                  current-activations
                                  next-activations)
                      (+ 1.0 plasticity))]
    (dtype/set-value! weights
                      [current-activations
                       next-activations]
                      w))
  weights)

(defn hebbian-plasticity
  [{:as state :keys [activations]}]
  (assoc state
         :weights
         (hebbian-plasticity-1
          (assoc state
                 :current-activations
                 (peek (:activation-history state))
                 :next-activations activations))))

(defn cap-k-1 [k synaptic-input]
  (bitmap/->bitmap (take k (argops/argsort > synaptic-input))))

(defn cap-k [{:keys [cap-k-k synaptic-input] :as state}]
  (assoc state :activations (cap-k-1 cap-k-k synaptic-input)))



;;
;; Dale's law (Eccles, 1964) states that a neuron can only excite or inhibit its target neurons, but not both.
;; Our inhibition model is in principle modeling a population of inhibitory neurons.
;;
;;
;; Imagine [I], the population of inhibitory neurons, and [E], the population of excitatory neurons.

;;                                P
;;                                | (input from sensors etc.)
;;                                |
;; +--------+ <---------  +-------+-+
;; |        |             |       v |
;; |   I    |----------|  |   E     |
;; +--------+             +---------+
;;   ^
;;   |
;;   |
;; inhibitory-drivers
;;
;;
;;
;; We make I a little slugish to change, producing a hysteresis effect.
;;
;;
;; case 0):
;; Some steady state where I(t), the proportion of inhibitory neurons,
;; and E(t), the proportion of excitatory neurons are balancing, so that E(t) count stays at some medium level

;; case 1):
;; A lot of P comming in, E is high for a moment, activating I in turn, now this can oscillitate (depends on the dynamics I guess)
;; until it goes back to 0).
;;
;;
;; case 2)
;; A lot of inhibitory-drivers comming in,
;; perhaps now it too oscillates. E first being really low, then I is low again from that and so forth.
;;
;;
;; Basically, I wonder if it would make sense to have some kind of bouncy ball implementation for I.
;; So that it tends to go into some state but it also is allowed to oscillate around.
;;
;; case 1) is especially funny, because it would automatically make a Braitenberg thought pump.
;; A thought pump is the inhibition going up and down, finding best connected cell assemblies.
;;
;;

(defn threshold-inhibition-1 [threshold synaptic-input]
  (argops/argfilter #(< threshold %) synaptic-input))


;; Still thinking about the threshold-device.

;; =====================
;; threshold-device-inhibition
;; =====================
;;
;;
;; target-n: the number of neurons that should be active
;;
;; 1. calculate an error
;; 2. if error is positive, we are under the target, so we need to inhibit more
;; 3. if error is negative, we are over the target, so we need to inhibit less
;;
;; Error can be calculated:
;; - the difference between the target-n and the current number of active neurons
;; - the ratio between the target-n and the current number of active neurons
;;
;; The threshold is then updated by the error.
;;
;;
;; Note that `threshold` is a number which models the output of the inhibitory neurons.
;;
;;

(defn threshold-device-inhibition
  [{:as state
    :keys [threshold synaptic-input last-threshold target-n
           threshold-device-damping activations]}])



;;
;; attenuation
;; ----------------
;;
;; This is a 'excitability' concept on the sub-neuronal-area level.
;; In the simplest case this says that if I was active recently I am less eager to be active again.
;; Or to say it the other way around 'fresh' neurons have a high excitability.
;;
;; If you have a thought pump increasing and decreasing threshold
;; It is interesting to implement attenuation,
;; If I was active in the recent past, I am less eager to be active again.
;; This makes thought sequences, biasing towards jumping between cell assemblies.
;; In the limit you sort of see the activation flowing around, never staying.
;;
;;
;; From considering this I thought it would make sense that language-cortex (wernicke?) has a high
;; attenuation, this way you get a succesion flow of language.
;; The other way around in a vision area, since objects just stay around.
;; A single enduring idea is useless in the realm of language, but useful in the realm of vision.
;;
;; This is similar to G. Palms 'prediction areas' and 'completion areas'
;; Millers 'plan' and 'image'
;;
;; We can implement this by keeping around the activation states a little, then apply attenuation
;; as a malus to the synaptic input. (The more I was active recently, the less eager I am for more activation)
;;
;;

;; ------------------------------
;; 'Summed History' Attenuation model
;; ------------------------------
;;
;;    1. Sum history
;;
;;      activation history
;;     +----+ +----+ +----+
;;     |    | |    | |    |
;;     | -X-+-+--X-+-+----+--->
;;     |    | |    | |    |
;;     | -X-+-+--X-+-+--X-+--->   how-often-active * attenuation-malus
;;     |    | |    | |    |
;;     +----+ +----+ +----+                       |
;;                                                |
;;     <-|----|------|------                      |
;;          n-hist                                |
;;                          +---------------------+
;;                          |
;;  2. apply malus          | + 1, so we don't divide by less than 1
;;                          |
;;                          v
;;     +----+            +----+
;; n   | 0  |            | .. |
;; n2  | 1.0|            | 0.3|
;; n3  | 11.|      /     | 2. |    ------------>  updated synaptic input
;;     |    |            |    |
;;     |    |            |    |
;;     |    |            |    |
;;     +----+            +----+
;;     synaptic input      attenuation malus
;;
;;
;;
;; The malus can be applied:
;;
;; - substractive
;; - divisive (like this). (thining absolutely with changing inputs is harder)
;;
;;
;; The malus can be determined:
;;
;; - sum (like this)
;; - weighted sum (for instance 'more recent, more attentuation')
;; - ..
;;
;; You can also decide that the malus is somehow %-wise of the current inputs
;; Becuase with the plasticity we always have to keep in mind that inputs are
;; only make sense to compare within a time step.
;;
;; attenuation-malus
;;
;; 0: no attenuation
;;
;; between 0 and 1
;;
;; Exactly 1.0: Half the synaptic input after being active 2 and so forth
;;
;; I'll just say that attenuation-hist-n is something like 10 and factor is something like 0.1
;;

(defn attenuation
  [{:as state
    :keys [attenuation-malus synaptic-input
           attenuation-hist-n activation-history]}]
  (update state
          :synaptic-input
          (fn [input]
            (time
             (dtype-fn//
              input
              (-> (dtt/reduce-axis
                   (into
                    []
                    (map (fn [indices]
                           (dtt/compute-tensor
                            (some-fn
                             (bitmap/bitmap-value->map
                              indices
                              1)
                             (constantly 0))
                            :int32))
                         (take attenuation-hist-n
                               activation-history)))
                   dtype-fn/sum
                   0)
                  (dtype-fn/* attenuation-malus)
                  (dtype-fn/+ 1.0)))))))


;; -------------------------
;; intrinsic excitability plasticity
;; -------------------------


;; ================
;; intrinsic excitability plasticity
;; ================
;;
;; https://faster-than-light-memes.xyz/biological-notes-on-the-vehicles-cell-assemblies.html
;; *Iceberg Cell Assemblies*
;;
;; Paper: https://www.biorxiv.org/content/10.1101/2020.07.29.223966v1
;;
;;
;; quote:
;; we find increases in neuronal excitability, accompanied by increases in membrane resistance and a reduction in spike threshold. We conclude that the formation of neuronal ensemble by photostimulation is mediated by cell-intrinsic changes in excitability, rather than by Hebbian synaptic plasticity or changes in local synaptic connectivity. We propose an “iceberg” model, by which increased neuronal excitability makes subthreshold connections become suprathreshold, increasing the functional effect of already existing synapses and generating a new neuronal ensemble.
;;___
;;
;;
;; So instead of hebbian plasticity, we can try to model a cell-intrinsic `excitability`, `intrinsic-excitability`.
;;
;; The easiest model coming to mind is the same as the attenuation above. (but inverted).
;; A cummulative excitability, with a decay.
;;
;;
;; `excitability-growth`: Could have been called learning rate, to make it sound like machine learning
;;
;; `excitability-decay`: Relative decay each time step
;;
;; `excitabilities`: You could imagine pre-allocating this, probably this is somewhat random in biology.
;;
;;
;; 0. Grow excitability for all active neurons, (add excitability-growth)
;; 1. Decay excitabilities multiplicatively by excitability-decay
;; 2. excitabilities multiplicatively on the sum of the inputs for each neuron.
;;
;;
;; 0-2 could also be more complecated functions
;;
;; An excitability of 0.0 means your inputs are at baseline.
;; An excitability of 1.0 means your inputs count double and so forth.
;; In principle, negative numbers are allowed. (but not utilized by this step model here).
;; In this case this would flip into an attenuation or depression model.
;;
;; -1.0 is the minimal meaningful number, saying that synaptic input is 0, beyond that you would
;; get negative synaptic-inputs, which are not defined behaviour.
;;
;;

(defn intrinsic-excitability
  [{:as state
    :keys [excitabilities excitability-growth
           excitability-decay synaptic-input n-neurons
           activations]}]
  (let [excitabilities (or excitabilities
                           (dtt/compute-tensor [n-neurons]
                                               (constantly
                                                 0)
                                               :float32))
        ;; decay excitabilities
        excitabilities (dtype-fn/* excitabilities
                                   (- 2 excitability-decay))
        excitabilities
          ;; accumulate the excitabilities on everybody
          ;; active
          (dtype-fn/+ excitabilities
                      (dtt/compute-tensor
                        [n-neurons]
                        (some-fn (bitmap/bitmap-value->map
                                   activations
                                   excitability-growth)
                                 (constantly 0))
                        :float32))]
    (assoc state
      ;; if you have excitability, your inputs count
      ;; more
      :synaptic-input (dtype-fn/*
                        synaptic-input
                        (dtype-fn/+ 2 excitabilities))
      :excitabilities excitabilities)))

(defn update-neuronal-area
  [{:as state
    :keys [activations activation-history weights
           inhibition-model plasticity-model
           history-depth]}]
  (let [update-hist
        ;; hist is a short seq (history-depth) of bitmaps currently
        (fn [hist actv]
          (into []
                (take-last history-depth (conj hist actv))))
        synaptic-input (synaptic-input weights activations)]
    (cond->
        state
        :hist (update :activation-history update-hist activations)
        :input (assoc :synaptic-input synaptic-input)
        inhibition-model (inhibition-model)
        plasticity-model (plasticity-model))))



(defn append-activations
  [state inputs]
  (update state
          :activations
          (fn [activations]
            (bitmap/reduce-union [activations inputs]))))

(defn read-activations [state]
  (:activations state))

(defn rand-projection [n k]
  (bitmap/->bitmap
   (repeatedly k #(rand-int n))))

;; n ~ 1e7
;; k ~ 1e3  (square root of n)
;; p ~ 1e-3 (density)
;; β ~ 1e-1

(defn ->neuronal-area
  [n-neurons]
  {:activation-history []
   :activations #{}
   :cap-k-k (* 0.1 n-neurons)
   :history-depth 10
   :inhibition-model cap-k
   :plasticity 0.1
   :plasticity-model hebbian-plasticity
   :weights (->random-directed-graph n-neurons 0.1)})

(defn rand-stimulus [n n-neurons]
  (rand-projection n-neurons n))

;; (defn ->projection
;;   [k n density]
;;   (dtt/compute-tensor
;;    [k n]
;;    (fn [_ _] (< (fm.rand/frand) density))
;;    :boolean))

(defn ->projection
  [k n density]
  (into
   []
   (repeatedly k #(rand-projection n (* density n)))))

(comment

  (time
   (->
    (append-activations
     (->neuronal-area 1000)
     (rand-stimulus 100 1000))
    (update-neuronal-area)
    (append-activations
     (rand-stimulus 100 1000))
    (update-neuronal-area)
    (update-neuronal-area)
    (update-neuronal-area)
    (update-neuronal-area)))

  (def s1
    (time
     (->
      (append-activations
       (->neuronal-area 1000)
       [(rand-stimulus 100 1000)])
      (update-neuronal-area)
      (update-neuronal-area))))

  ;; -> ~E
  ;; (let [exitation
  ;;       (count
  ;;        (bitmap/reduce-intersection
  ;;         [(rand-projection 1000 100)
  ;;          (read-activations state)]))
  ;;       ;; that is ~2% of the neurons
  ;;       goal 2]
  ;;   )

  ;; (dtype-fn/or
  ;;  [false false]
  ;;  [true false]
  ;;  [false true])

  (time
   (def train-states
     (doall
      (let [random-stimulus (rand-stimulus 50 1000)
            state (->neuronal-area 1000)
            state
            (assoc
             state
             :inhibition-model
             (let [last-threshold (atom 0)]
               (fn [state synaptic-input]
                 (def synaptic-input-1 synaptic-input)
                 (def state state)
                 (let error
                   [excitation
                    (dtype/ecount (read-activations
                                   state)) goal 100
                    error (- goal excitation) last-v
                    (or @last-threshold
                        ;; In biology, this
                        ;; would need to
                        ;; finetune itself in
                        ;; feedback loops etc.
                        ;; If we don't yet have
                        ;; a threshold, we
                        ;; simply try with some
                        ;; guess and see what
                        ;; happens a mean
                        ;; threshold would mean
                        ;; roughtly 50% active
                        ;;
                        ((dtype-fn/mean-fast
                          synaptic-input)))])
                 (cap-k 100 synaptic-input))))]
        (reductions (fn [state r-state]
                      (assoc (update-neuronal-area-1
                              (append-activations state
                                                  r-state))
                             :inputs r-state))
                    state
                    (repeat 20 random-stimulus)))))))




(comment

  (def mystate
    {:activations
     (into #{} (repeatedly 200 #(rand-int 1000)))
     ;; (->neurons 1000)
     :weights (->random-directed-graph 1000 0.1)
     :inhibition-model
     (fn [_ synaptic-input]
       (cap-k 100 synaptic-input))
     :plasticity 0.1
     :plasticity-model hebbian-plasticity})


  (time
   (def train-states
     (doall
      (let
          [random-stimulus (rand-stimulus 150 1000)
           state (->neuronal-area 1000)]
          (reductions
           (fn [state r-state]
             (assoc
              (update-neuronal-area
               (append-activations
                state
                r-state))
              :inputs r-state))
           state
           (repeat 20 random-stimulus))))))

  (def stimulus-a (rand-stimulus 150 1000))
  (def stimulus-b (rand-stimulus 150 1000))

  (def state (->neuronal-area 1000))

  ;; state with a cell assembly A
  (def state-with-a
    (reduce
     (fn [state i] (update-neuronal-area (append-activations state stimulus-a)))
     state
     (range 5)))

  (def state-with-and-b
    (reduce
     (fn [state i] (update-neuronal-area (append-activations state stimulus-b)))
     state-with-a
     (range 5)))

  ;; so is A and B now associated?

  ;; if I give you A, do you pattern complete to something that is a little bit like B?

  ( state-with-and-b)

  (dtype/ecount
   (bitmap/reduce-intersection
    [(read-activations
      (update-neuronal-area
       (append-activations state-with-and-b stimulus-a)))
     ;; these is basically B
     (read-activations state-with-and-b)]))

  59


  ;; so A and B overlap by something like 1/2 k

  (dtype/ecount
   (bitmap/reduce-intersection
    [(read-activations
      (update-neuronal-area
       (update-neuronal-area
        (append-activations state-with-and-b stimulus-a))))
     ;; these is basically B
     (read-activations state-with-and-b)]))
  76


  ;; oh, the overlap is bigger on the second time step

  (dtype/ecount
   (bitmap/reduce-intersection
    [(read-activations
      (update-neuronal-area
       (update-neuronal-area
        (update-neuronal-area
         (append-activations state-with-and-b stimulus-a)))))
     ;; these is basically B
     (read-activations state-with-and-b)]))
  77

  ;; and slightly bigger on the third

  ;; observation:
  ;; You can present such a network stimulus A 5 times, then sitmulus B 5 times
  ;; It already has the association A->B
  ;; Actually you get the same numbers with 'state'
  ;;

  (dtype/ecount
   (bitmap/reduce-intersection
    [(read-activations
      (update-neuronal-area
       (append-activations state-with-and-b stimulus-b)))
     ;; these is basically B
     (read-activations state-with-and-b)]))





  (dtype/ecount
   (bitmap/reduce-intersection
    [(read-activations (update-neuronal-area
                        (append-activations
                         state-with-and-b
                         stimulus-b)))
     (read-activations (update-neuronal-area
                        (append-activations
                         state-with-and-b
                         stimulus-a)))]))

  (dtype/ecount
   (bitmap/reduce-intersection
    [
     ;; -> B
     (read-activations
      (update-neuronal-area
       (append-activations
        state-with-and-b
        stimulus-b)))
     (read-activations
      ;; After A is created, the next step will produce basically B
      (update-neuronal-area
       ;; -> A
       (update-neuronal-area
        (append-activations
         state-with-and-b
         stimulus-a))))]))

  (dtype/ecount
   (bitmap/reduce-intersection
    [
     (read-activations
      (update-neuronal-area
       (update-neuronal-area
        (update-neuronal-area
         ;; -> B
         (append-activations
          state-with-and-b
          stimulus-b)))))
     (read-activations
      ;; After A is created, the next step will produce basically B
      (update-neuronal-area
       ;; -> A
       (update-neuronal-area
        (append-activations
         state-with-and-b
         stimulus-a))))]))

  ;; And giving him B says B,
  ;; Then A -> B
  ;; But B is sort of stable

  ;; So I think we can get from the network the info that A is after B,
  ;; they are together
  ;; But B is like the destination, and A a possible source


  )

;; ========================
;; attenuation
;; =================
(comment
  (let [n-neurons 3]
    (attenuation
     (update-neuronal-area
      {:activation-history
       [(bitmap/->bitmap
         #{0 2 2})]
       :activations #{0 2 2}
       :attenuation-epsilon 1e-8
       :attenuation-hist-n 1
       :attenuation-malus 2.0
       :cap-k-k 10
       :inhibition-model cap-k
       :history-depth 10
       :plasticity 0.1
       :plasticity-model hebbian-plasticity
       :weights (->random-directed-graph n-neurons 0.1)})))


  (let [n-neurons 100]
    (def mystate
      {:activation-history []
       :activations
       (into #{} (repeatedly 10 #(rand-int n-neurons)))
       :attenuation-epsilon 1e-8
       :attenuation-hist-n 1
       :attenuation-malus 2.0
       :cap-k-k 10
       :history-depth 10
       :inhibition-model cap-k
       :plasticity 0.1
       :plasticity-model hebbian-plasticity
       :weights (->random-directed-graph n-neurons 0.1)})
    (let [state
          (-> (update-neuronal-area mystate)
              (update-neuronal-area)
              (update-neuronal-area))]
      [state
       (attenuation state)])))



;; ===================
;; Threshold device
;; ===================
;; wip




;; intrinsic excitability

(comment
  (def excitability-decay 0.1)
  (def activations (bitmap/->bitmap #{0 2 2}))
  (def n-neurons 100)
  (def excitability-growth 0.025)
  (def synaptic-input (dtt/->tensor (range n-neurons)))

  (let [update-na
        (fn [{:as state
              :keys [activations weights inhibition-model
                     history-depth]}]
          (let [synaptic-input (synaptic-input weights
                                               activations)]
            (cond-> state
              :input (assoc :synaptic-input synaptic-input)
              inhibition-model (inhibition-model))))
        n-neurons 1000
        state {:activations #{}
               :cap-k-k (* 0.1 n-neurons)
               :excitability-decay 0.1
               ;; disabling it
               :excitability-growth 0.0
               :inhibition-model (comp cap-k
                                       intrinsic-excitability)
               :n-neurons n-neurons
               :weights (->random-directed-graph n-neurons
                                                 0.1)}
        stimulus-a (rand-stimulus 100 n-neurons)
        update-1 (fn [state stimulus]
                   (-> state
                       (append-activations stimulus)
                       update-na))]
    [(dtype/ecount (bitmap/reduce-intersection
                    (let [trained (reduce update-1
                                          state
                                          (repeat 5 stimulus-a))]
                      [(:activations trained)
                       (:activations
                        (update-1 trained stimulus-a))])))
     (dtype/ecount (bitmap/reduce-intersection
                    (let [trained (reduce update-1
                                          state
                                          (repeat 5 stimulus-a))]
                      [(:activations trained)
                       (:activations
                        (update-na
                         (update-na
                          (update-1
                           trained
                           (take 50 stimulus-a)))))])))
     (dtype/ecount (bitmap/reduce-intersection
                    (let [trained (reduce update-1
                                          state
                                          (repeat 5 stimulus-a))]
                      [(:activations trained)
                       (:activations
                        (update-na
                         (update-na
                          (update-1
                           trained
                           (take 50
                                 (rand-stimulus
                                  100
                                  n-neurons))))))])))])

  [51 20 17]
  ;; -> without excitability plasticy a random stimulis is not different from the one the network was trained on

  (let
      [update-na
       (fn [{:as state
             :keys [activations weights inhibition-model
                    history-depth]}]
         (let [synaptic-input (synaptic-input weights
                                              activations)]
           (cond-> state
             :input (assoc :synaptic-input synaptic-input)
             inhibition-model (inhibition-model))))
       n-neurons 1000
       state {:activations #{}
              :cap-k-k (* 0.1 n-neurons)
              :excitability-decay 0.1
              :excitability-growth 0.2
              :inhibition-model (comp cap-k intrinsic-excitability)
              :n-neurons n-neurons
              :weights (->random-directed-graph n-neurons
                                                0.1)}
       stimulus-a (rand-stimulus 100 n-neurons)
       update-1 (fn [state stimulus]
                  (-> state
                      (append-activations stimulus)
                      update-na))]
      [(dtype/ecount (bitmap/reduce-intersection
                      (let [trained (reduce update-1
                                            state
                                            (repeat 5 stimulus-a))]
                        [(:activations trained)
                         (:activations
                          (update-1 trained stimulus-a))])))
       (dtype/ecount (bitmap/reduce-intersection
                      (let [trained (reduce update-1 state (repeat 5 stimulus-a))]
                        [(:activations trained)
                         (:activations
                          (update-na
                           (update-na
                            (update-1
                             trained
                             (take 50 stimulus-a)))))])))])
  [97 90]

  ;; hey hello cell assembly!
  ;; but is basically the same cell assembly also for random stimuli

  ;; played around with the excitability-growth
  ;; 0.1 -> 75
  ;; 0.2 -> 90
  ;; 0.3 -> 92
  ;; 0.4 -> 90



  (let [update-na
        (fn [{:as state
              :keys [activations weights inhibition-model
                     history-depth]}]
          (let [synaptic-input (synaptic-input weights
                                               activations)]
            (cond-> state
              :input (assoc :synaptic-input synaptic-input)
              inhibition-model (inhibition-model))))
        n-neurons 1000
        state {:activations #{}
               :cap-k-k (* 0.1 n-neurons)
               :excitability-decay 0.1
               :excitability-growth 0.3
               :inhibition-model (comp cap-k
                                       intrinsic-excitability)
               :n-neurons n-neurons
               :weights (->random-directed-graph n-neurons
                                                 0.1)}
        stimulus-a (rand-stimulus 100 n-neurons)
        update-1 (fn [state stimulus]
                   (-> state
                       (append-activations stimulus)
                       update-na))
        stimulus-b (rand-stimulus 100 n-neurons)]
    [
     ;; (let [trained
     ;;       (reduce update-1 state (repeat 5 stimulus-a))]
     ;;   [(dtype/ecount
     ;;     (bitmap/reduce-intersection
     ;;      [(:activations trained)
     ;;       (:activations
     ;;        (update-na (assoc trained
     ;;                          :activations
     ;;                          (into #{}
     ;;                                (take 50
     ;;                                      (rand-stimulus
     ;;                                       100
     ;;                                       n-neurons))))))]))
     ;;    (dtype/ecount
     ;;     (bitmap/reduce-intersection
     ;;      [(:activations trained)
     ;;       (:activations
     ;;        (update-na
     ;;         (assoc trained
     ;;                :activations
     ;;                (into #{} (take 50 stimulus-a)))))]))])

     ;; (let [trained
     ;;       (reduce update-1 state (repeat 5 stimulus-a))]
     ;;   [(dtype/ecount
     ;;     (bitmap/reduce-intersection
     ;;      [(:activations trained)
     ;;       (:activations
     ;;        (reduce update-1 state (repeat 5 (rand-stimulus 100 n-neurons))))]))
     ;;    (dtype/ecount
     ;;     (bitmap/reduce-intersection
     ;;      [(:activations trained)
     ;;       (:activations
     ;;        (reduce update-1 state (repeatedly 5 #(rand-stimulus 100 n-neurons))))]))
     ;;    ])

     (let [trained (reduce update-1 state (repeat 10 stimulus-a))
           trained (reduce update-1 trained (repeat 10 stimulus-b))
           ->cell-assembly
           (fn [state n stimulus]
             (last (take n (iterate update-na (assoc state :activations stimulus)))))]
       [(dtype/ecount
         (bitmap/reduce-intersection
          [(:activations (->cell-assembly trained 2 stimulus-a))
           (:activations (->cell-assembly trained 2 (into #{} (take 50 stimulus-a))))]))
        (dtype/ecount
         (bitmap/reduce-intersection
          [(:activations (->cell-assembly trained 2 stimulus-b))
           (:activations (->cell-assembly trained 2 (into #{} (take 50 stimulus-b))))]))
        (dtype/ecount
         (bitmap/reduce-intersection
          [(:activations (->cell-assembly trained 2 (into #{} (take 50 stimulus-b))))
           (:activations (->cell-assembly trained 2 (into #{} (take 50 stimulus-a))))]))
        (dtype/ecount
         (bitmap/reduce-intersection
          [(:activations (->cell-assembly trained 2 (into #{} (take 50 stimulus-b))))
           (:activations (->cell-assembly trained 2 (into #{} (take 50 (rand-stimulus 100 n-neurons)))))]))
        (dtype/ecount
         (bitmap/reduce-intersection
          [(:activations (->cell-assembly trained 2 (into #{} (take 50 stimulus-b))))
           (:activations (->cell-assembly trained 5 (into #{} (take 50 (rand-stimulus 100 n-neurons)))))]))])])
  [[86 79 56 58 65]]


  ;; association A - B
  (let [update-na
        (fn [{:as state
              :keys [activations weights inhibition-model
                     history-depth]}]
          (let [synaptic-input (synaptic-input weights
                                               activations)]
            (cond-> state
              :input (assoc :synaptic-input synaptic-input)
              inhibition-model (inhibition-model))))
        n-neurons 1000
        state {:activations #{}
               :cap-k-k (* 0.1 n-neurons)
               :excitability-decay 0.1
               :excitability-growth 0.3
               :inhibition-model (comp cap-k
                                       intrinsic-excitability)
               :n-neurons n-neurons
               :weights (->random-directed-graph n-neurons
                                                 0.1)}
        stimulus-a (rand-stimulus 100 n-neurons)
        update-1 (fn [state stimulus]
                   (-> state
                       (append-activations stimulus)
                       update-na))
        stimulus-b (rand-stimulus 100 n-neurons)]
    [(let [trained
           (reduce update-1 state
                   (repeat 5 (bitmap/reduce-union [stimulus-a stimulus-b])))
           ->cell-assembly (fn [state n stimulus]
                             (last (take n
                                         (iterate
                                          update-na
                                          (assoc state
                                                 :activations
                                                 stimulus)))))]
       [(dtype/ecount
         (bitmap/reduce-intersection
          [(:activations (->cell-assembly trained 2 (into #{} (take 50 stimulus-b))))
           (:activations (->cell-assembly trained 2 (into #{} (take 50 (rand-stimulus 100 n-neurons)))))]))
        (dtype/ecount
         (bitmap/reduce-intersection
          [(:activations (->cell-assembly trained 2 (into #{} (take 50 stimulus-b))))
           (:activations (->cell-assembly trained 2 stimulus-b))]))
        (dtype/ecount
         (bitmap/reduce-intersection
          [(:activations (->cell-assembly trained 2 (into #{} (take 50 stimulus-a))))
           (:activations (->cell-assembly trained 2 stimulus-a))]))])])




  ;; association A -> B
  (let [update-na
        (fn [{:as state
              :keys [activations weights inhibition-model
                     history-depth]}]
          (let [synaptic-input (synaptic-input weights
                                               activations)]
            (cond-> state
              :input (assoc :synaptic-input synaptic-input)
              inhibition-model (inhibition-model))))
        n-neurons 1000
        state {:activations #{}
               :cap-k-k (* 0.1 n-neurons)
               :excitability-decay 0.1
               :excitability-growth 0.3
               :inhibition-model (comp cap-k
                                       intrinsic-excitability)
               :n-neurons n-neurons
               :weights (->random-directed-graph n-neurons
                                                 0.1)}
        stimulus-a (rand-stimulus 100 n-neurons)
        update-1 (fn [state stimulus]
                   (-> state
                       (append-activations stimulus)
                       update-na))
        stimulus-b (rand-stimulus 100 n-neurons)]
    [(let [trained
           (reduce update-1 state
                   (take 10
                         (cycle [stimulus-a stimulus-b])))
           ->cell-assembly (fn [state n stimulus]
                             (last (take n (iterate update-na (assoc state :activations stimulus)))))])])





  )
