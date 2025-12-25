#import "graph.typ": *
#import "style.typ": *

#show: main-template

#set align(center)
#let graphspace = 3.5em


#let marshalls-study = align(center)[
    #v(1em)
    #show math.equation: set text(size: 0.65em, fill: colors.math)
    #show math.equation: set par(leading: 0.2em, spacing:0em)
    #let x = $bot #[At Marshalls,]$
    #let xi = $bot #[At Marshalls, \ grief]$
    #let xia = $bot #[At Marshalls, \ grief washes away]$
    #let xian = $bot #[At Marshalls, grief washes away \ when families see justice served.] top$
    #let xiam = $bot #[At Marshalls, grief washes away \ as the sun bathes her naked trans body.] top$
    #let xib = $bot #[At Marshalls, grief hides \ in the clearance aisle.] top$
    #let xk = $bot #[At Marshalls, \ silence follows]$
    #let xka = $bot #[At Marshalls, silence follows, \ me olvidé my rewards card.] top$
    #let xkb = $bot #[At Marshalls, silence follows \ as the last domtop leaves the beach.] top$
    #diagram(
    // make the grid tighter
    spacing: (15mm, 5mm),
    cell-size: (0pt, 0pt),
    // make nodes/labels more compact
    node-inset: 4pt,
    // keep lines/marks proportionate at the smaller size
    edge-stroke: 0.75pt,
    mark-scale: 50%,
    {

    node(pos:(0, 0), name: <x>, label:$#x$)

    node(pos:(0.65, -0.9), name: <xi>, label:$#xi$)
    node(pos:(1., 1.3), name: <xk>, label:$#xk$)

    node(pos:(1.54, -1.85), name: <xib>, label:$#xib$)
    node(pos:(2, -0.85), name: <xia>, label:$#xia$)
    
    node(pos:(2, 0.5), name: <xka>, label:$#xka$)
    node(pos:(2.5, 1.5), name: <xkb>, label:$#xkb$)

    node(pos:(3, -1.85), name: <xian>, label:$#xian$)
    node(pos:(3, -0.55), name: <xiam>, label:$#xiam$)


    edge(<x>, <xi>, "->", label:$0.1$, bend: +5deg,  label-side: center,  label-fill: true)
    edge(<x>, <xk>, "->", label:$0.3$, bend: -5deg,  label-side: center,  label-fill: true)

    edge(<xi>, <xib>, "->", label:$0.7$, bend: +5deg,  label-side: center,  label-fill: true)
    edge(<xi>, <xia>, "->", label:$0.05$, bend: -5deg,  label-side: center,  label-fill: true)

    edge(<xk>, <xka>, "->", label:$0.00002$, bend: +5deg,  label-side: center,  label-fill: true)
    edge(<xk>, <xkb>, "->", label:$0.99$, bend: -5deg,  label-side: center,  label-fill: true)

    edge(<xia>, <xian>, "->", label:$0.001$, bend: +5deg,  label-side: center,  label-fill: true)
    edge(<xia>, <xiam>, "->", label:$0.4$, bend: -5deg,  label-side: center,  label-fill: true)

  })
]
#figure(
  marshalls-study
)


#let C_gb  = rgb(77, 135, 219)
#let C_ds  = rgb(237, 125, 33)
#let C_sad = rgb(84, 168, 84)

#let marshalls-prompt = align(center)[
  #v(2em)

  #let edge-size = 0.65em
  #show math.equation: set par(leading: 0.0em, spacing: 0.25em)
  
  // ── Dimension colors (used everywhere)
  // Order: [gay beach, department store, sadness]
  
  #let node_label(
    pa, pb, pc, oa, ob, oc,
    digits: 3,
    fade: 40%,        // transparency for brackets/subscripts
    gap: 0.25em,      // spacing between numbers
    label_size: 0.85em
  ) = [
    #let faded = rgb(0, 0, 0, fade)
    #show math.equation: set text(fill: faded, size:label_size)
    #set text(size:label_size, weight: "black")
    $[
      #text(fill: C_gb)[#calc.round(pa, digits: digits)]
      #h(gap)
      #text(fill: C_ds)[#calc.round(pb, digits: digits)]
      #h(gap)
      #text(fill: C_sad)[#calc.round(pc, digits: digits)]
    ]_phi$
    $[
      #text(fill: C_gb)[#calc.round(oa, digits: digits)]
      #h(gap)
      #text(fill: C_ds)[#calc.round(ob, digits: digits)]
      #h(gap)
      #text(fill: C_sad)[#calc.round(oc, digits: digits)]
    ]_omega$
    
  ]
