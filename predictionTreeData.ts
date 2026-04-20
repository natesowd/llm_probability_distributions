export interface PredictionNode {
  word: string;
  prob: number;
  children: PredictionNode[];
  endProb: number;
}

const n = (word: string, prob: number, children: PredictionNode[] = [], endProb = 0): PredictionNode =>
  ({ word, prob, children, endProb });

export const predictionTree: PredictionNode = n("European Union", 1, [
  // ════════════════════════════════════════════════════════════
  // BRANCH 1: Reaches (0.446719)
  // ════════════════════════════════════════════════════════════
  n("Reaches", 0.446719, [
    n("AI", 0.592304, [
      n("Ethics", 0.92219, [
        n("Consensus", 0.362775, [
          n("Agreement", 0.534971),
          n("Framework", 0.17368),
          n("Deal", 0.063893),
          n("Finally", 0.060022)
        ]),
        n("Landmark", 0.117757, [
          n("Agreement", 0.500022),
          n("Accord", 0.303279),
          n("Deal", 0.143259),
          n("Pact", 0.019388)
        ]),
        n("Framework", 0.103943, [
          n("Agreement", 0.470407),
          n("Deal", 0.222205),
          n("Accord", 0.118938),
          n("Consensus", 0.104955)
        ]),
        n("Agreement", 0.055637, [
          n("Milestone", 0.201853),
          n("Framework", 0.182924),
          n("Finally", 0.076254)
        ], 0.182924)
      ]),
      n("Governance", 0.01689, [
        n("Milestone", 0.362019, [
          n("Agreement", 0.507865),
          n("Finally", 0.11332),
          n("Today", 0.03679)
        ], 0.088254),
        n("Agreement", 0.151934, [
          n("Finally", 0.013226),
          n("Today", 0.004294)
        ], 0.937545),
        n("Landmark", 0.134065, [
          n("Deal", 0.478369),
          n("Agreement", 0.175982),
          n("Accord", 0.06474)
        ], 0.225966),
        n("Consensus", 0.118322, [
          n("Agreement", 0.159397),
          n("Finally", 0.045668),
          n("Deal", 0.024444)
        ], 0.630429)
      ]),
      n("Morality", 0.013118, [
        n("Consensus", 0.303281),
        n("Milestone", 0.142946),
        n("Accord", 0.12643, [
          n("Finally", 0.076124),
          n("Agreement", 0.049149),
          n("Framework", 0.012427)
        ], 0.678486),
        n("Framework", 0.076683, [
          n("Agreement", 0.363402),
          n("Deal", 0.220415),
          n("Accord", 0.071558)
        ], 0.194515)
      ]),
      n("Ethic", 0.005801, [
        n("Consensus", 0.460131),
        n("Guidelines", 0.085166, [
          n("Agreement", 0.311797),
          n("Consensus", 0.311797),
          n("Deal", 0.069571),
          n("Framework", 0.050899)
        ]),
        n("Standards", 0.070605, [
          n("Deal", 0.362591),
          n("Agreement", 0.282386),
          n("Accord", 0.071398)
        ], 0.103884),
        n("Framework", 0.04023, [
          n("Agreement", 0.490175),
          n("Deal", 0.180325),
          n("Consensus", 0.123936),
          n("Accord", 0.109373)
        ])
      ])
    ]),
    n("Landmark", 0.279783, [
      n("AI", 0.995054, [
        n("Ethics", 0.831477, [
          n("Accord", 0.601297),
          n("Deal", 0.172274),
          n("Agreement", 0.081377),
          n("Consensus", 0.063376)
        ]),
        n("Accord", 0.041397, [
          n("Agreement", 0.181348),
          n("Deal", 0.062672),
          n("Finally", 0.019114)
        ], 0.594619),
        n("Morality", 0.024511),
        n("Agreement", 0.019554, [
          n("Deal", 0.088026),
          n("Framework", 0.039061),
          n("Settlement", 0.039061)
        ], 0.611022)
      ]),
      n("Artificial", 0.004608, [
        n("Agreement", 0.577174, [
          n("Deal", 0.003212),
          n("Settlement", 0.001615)
        ], 0.989764),
        n("Intelligence", 0.272638, [
          n("Accord", 0.625254),
          n("Deal", 0.158089),
          n("Pact", 0.139513),
          n("Agreement", 0.035274)
        ]),
        n("Ethics", 0.060834, [
          n("Deal", 0.46066),
          n("Accord", 0.406531),
          n("Pact", 0.062343),
          n("Agreement", 0.025989)
        ]),
        n("Consensus", 0.028728)
      ]),
      n("Agreement", 8.4e-05, [
        n("on", 0.901282, [
          n("AI", 0.925762),
          n("Ethics", 0.067062),
          n("A", 0.001787),
          n("Artificial", 0.001392)
        ]),
        n("AI", 0.009681, [
          n("Ethics", 0.249499),
          n("Guidelines", 0.086225),
          n("Standards", 0.067152)
        ], 0.117855),
        n("with", 0.005872, [
          n("AI", 0.884597),
          n("Tech", 0.05655),
          n("Industry", 0.016202),
          n("Reg", 0.011854)
        ])
      ]),
      n("Tech", 5.1e-05, [
        n("Ethics", 0.587119, [
          n("Accord", 0.606123),
          n("Deal", 0.286312),
          n("Pact", 0.043907),
          n("Agreement", 0.030177)
        ]),
        n("Accord", 0.102026, [
          n("Agreement", 0.064188),
          n("Deal", 0.041443)
        ], 0.758413),
        n("Governance", 0.054611, [
          n("Deal", 0.639754),
          n("Accord", 0.183293),
          n("Pact", 0.086581),
          n("Agreement", 0.036092)
        ]),
        n("Morality", 0.0252)
      ])
    ]),
    n("Historic", 0.042906, [
      n("AI", 0.996338, [
        n("Ethics", 0.69883, [
          n("Accord", 0.601749),
          n("Consensus", 0.118489),
          n("Agreement", 0.092281),
          n("Deal", 0.092281)
        ]),
        n("Accord", 0.057363, [
          n("Agreement", 0.449805),
          n("Deal", 0.060874),
          n("Settlement", 0.057186)
        ], 0.155448),
        n("Morality", 0.049777, [
          n("Accord", 0.522913),
          n("Pact", 0.132213),
          n("Pledge", 0.116678),
          n("Deal", 0.102968)
        ]),
        n("Governance", 0.050623, [
          n("Deal", 0.598398),
          n("Accord", 0.194271),
          n("Pact", 0.091767),
          n("Agreement", 0.04912)
        ])
      ]),
      n("Artificial", 0.003171, [
        n("Agreement", 0.474684, [
          n("Deal", 0.009656),
          n("Settlement", 0.006234)
        ], 0.965868),
        n("Intelligence", 0.369684, [
          n("Accord", 0.676393),
          n("Pact", 0.13319),
          n("Deal", 0.103728),
          n("Pledge", 0.033368)
        ]),
        n("Consensus", 0.064232, [
          n("Agreement", 0.180605),
          n("Deal", 0.124128)
        ], 0.651945),
        n("Ethics", 0.030346, [
          n("Accord", 0.503048),
          n("Deal", 0.345739),
          n("Pact", 0.060081),
          n("Pledge", 0.03613)
        ])
      ]),
      n("Tech", 0.00016, [
        n("Ethics", 0.326109, [
          n("Accord", 0.615718),
          n("Deal", 0.22651),
          n("Agreement", 0.050541),
          n("Pact", 0.030655)
        ]),
        n("Morality", 0.121409, [
          n("Accord", 0.394944),
          n("Deal", 0.239546),
          n("Pact", 0.186558),
          n("Pledge", 0.113153)
        ]),
        n("Accord", 0.119969, [
          n("Agreement", 0.109445),
          n("Deal", 0.058582),
          n("Settlement", 0.048566)
        ], 0.591656),
        n("Governance", 0.093432, [
          n("Deal", 0.608682),
          n("Accord", 0.17439),
          n("Pact", 0.105773),
          n("Pledge", 0.038239)
        ])
      ]),
      n("Agreement", 0.00011, [
        n("on", 0.954639, [
          n("AI", 0.962987, [
            n("Ethics", 0.211971),
            n("Governance", 0.028687),
            n("Regulations", 0.0174)
          ], 0.652918),
          n("Ethics", 0.02908, [
            n("Guidelines", 0.00095)
          ], 0.996189),
          n("Artificial", 0.003065, [
            n("Intelligence", 0.270345)
          ], 0.725444),
          n("A.I", 0.00097)
        ]),
        n("Over", 0.005538, [
          n("AI", 0.982598, [
            n("Ethics", 0.019464),
            n("Regulation", 0.004623)
          ], 0.950393),
          n("Ethics", 0.008501),
          n("Artificial", 0.00095, [
            n("Intelligence", 0.322714)
          ], 0.673273)
        ], 0.003127),
        n("AI", 0.004591, [
          n("Guidelines", 0.162414),
          n("Standards", 0.162414),
          n("Ethics", 0.126488, [
            n("Guidelines", 0.0009),
            n("Standards", 0.0007)
          ], 0.994882),
          n("Regulations", 0.067704)
        ])
      ])
    ]),
    n("Groundbreaking", 0.02291, [
      n("AI", 0.99732, [
        n("Ethics", 0.917309, [
          n("Accord", 0.590951),
          n("Deal", 0.191854),
          n("Pact", 0.070579),
          n("Agreement", 0.054967)
        ]),
        n("Morality", 0.021306),
        n("Ethical", 0.00613),
        n("Accord", 0.011547, [
          n("Agreement", 0.205908),
          n("Finally", 0.085835),
          n("Deal", 0.045944)
        ], 0.409498)
      ]),
      n("Artificial", 0.002472, [
        n("Intelligence", 0.503614, [
          n("Accord", 0.696802),
          n("Pact", 0.155477),
          n("Deal", 0.073442),
          n("Pledge", 0.027018)
        ]),
        n("Agreement", 0.392215, [
          n("Settlement", 0.007357),
          n("Deal", 0.006493)
        ], 0.956324),
        n("Ethics", 0.05308, [
          n("Accord", 0.605242),
          n("Deal", 0.196493),
          n("Pact", 0.092817),
          n("Pledge", 0.038692)
        ]),
        n("Consensus", 0.019515)
      ]),
      n("Tech", 4.5e-05, [
        n("Ethics", 0.39588, [
          n("Accord", 0.605707),
          n("Deal", 0.222827),
          n("Pact", 0.081973),
          n("Pledge", 0.038722)
        ]),
        n("Accord", 0.272084, [
          n("Agreement", 0.03689),
          n("Deal", 0.019746)
        ], 0.829445),
        n("Morality", 0.050757),
        n("Agreement", 0.041725, [
          n("Today", 0.015707),
          n("Deal", 0.015707)
        ], 0.847171)
      ]),
      n("Ethics", 3.1e-05, [
        n("Agreement", 0.322045, [
          n("Deal", 0.021794),
          n("Today", 0.012418),
          n("Settlement", 0.012418)
        ], 0.817798),
        n("Accord", 0.322045, [
          n("Finally", 0.020341),
          n("Today", 0.014882)
        ], 0.864451),
        n("AI", 0.19533, [
          n("Accord", 0.82864),
          n("Agreement", 0.060026),
          n("Deal", 0.03213),
          n("Pact", 0.022082)
        ]),
        n("Deal", 0.055963, [
          n("on", 0.048497),
          n("Finally", 0.037769),
          n("Quickly", 0.025959)
        ], 0.628918)
      ])
    ])
  ]),
  // ════════════════════════════════════════════════════════════
  // BRANCH 2: Unites (0.150062)
  // ════════════════════════════════════════════════════════════
  n("Unites", 0.150062, [
    n("on", 0.985566, [
      n("AI", 0.903088, [
        n("Ethics", 0.729421, [
          n("Framework", 0.665967),
          n("Guidelines", 0.131137),
          n("Standards", 0.131137),
          n("Regulations", 0.017747)
        ]),
        n("Ethical", 0.059555, [
          n("Standards", 0.343419),
          n("Guidelines", 0.343419),
          n("Framework", 0.236028),
          n("Ground", 0.017098)
        ]),
        n("Moral", 0.04663, [
          n("Guidelines", 0.432598),
          n("Standards", 0.231553),
          n("Framework", 0.180334),
          n("Code", 0.051666)
        ]),
        n("Morality", 0.028147, [
          n("Standards", 0.37134),
          n("Guidelines", 0.255218),
          n("Framework", 0.136608),
          n("Pledge", 0.050255)
        ])
      ]),
      n("Artificial", 0.084, [
        n("Intelligence", 0.986329, [
          n("Ethics", 0.354479),
          n("Standards", 0.147769),
          n("Guidelines", 0.115082)
        ], 0.069801),
        n("Ethics", 0.00967, [
          n("Framework", 0.398699),
          n("Standards", 0.310507),
          n("Guidelines", 0.166203)
        ], 0.017518),
        n("Intellect", 0.001021, [
          n("Ethics", 0.308424),
          n("Guidelines", 0.211977),
          n("Standards", 0.187069),
          n("Regulations", 0.068819)
        ]),
        n("AI", 0.00079, [
          n("Ethics", 0.197022, [
            n("Guidelines", 0.0009),
            n("Standards", 0.00085)
          ], 0.99631),
          n("Guidelines", 0.197022),
          n("Standards", 0.197022),
          n("Regulations", 0.173872)
        ])
      ]),
      n("Responsible", 0.002874, [
        n("AI", 0.946508, [
          n("Standards", 0.283292),
          n("Framework", 0.220628),
          n("Guidelines", 0.171825),
          n("Practices", 0.071627)
        ]),
        n("Artificial", 0.053404, [
          n("Intelligence", 0.99672),
          n("Intellect", 0.001002),
          n("Governance", 0.0002)
        ])
      ]),
      n("Robotic", 0.002107, [
        n("AI", 0.394218, [
          n("Ethics", 0.355661),
          n("Standards", 0.190372),
          n("Guidelines", 0.148262),
          n("Regulations", 0.101899)
        ]),
        n("Ethics", 0.347896, [
          n("Standards", 0.397898),
          n("Framework", 0.273471),
          n("Guidelines", 0.187954),
          n("Principles", 0.013615)
        ]),
        n("Accountability", 0.060455, [
          n("Standards", 0.53488),
          n("Guidelines", 0.119348),
          n("Framework", 0.105324)
        ], 0.072388),
        n("Responsibility", 0.047083, [
          n("Standards", 0.373004),
          n("Guidelines", 0.256362),
          n("Framework", 0.155491),
          n("Rules", 0.036932)
        ])
      ])
    ]),
    n("Over", 0.003971, [
      n("AI", 0.950249, [
        n("Ethics", 0.548925, [
          n("Framework", 0.515243),
          n("Guidelines", 0.214785),
          n("Standards", 0.14762),
          n("Regulations", 0.029068)
        ]),
        n("Ethical", 0.082824, [
          n("Standards", 0.35486),
          n("Guidelines", 0.35486),
          n("Framework", 0.215234),
          n("Principles", 0.012143)
        ]),
        n("Regulation", 0.051058, [
          n("Framework", 0.210491),
          n("Guidelines", 0.144668),
          n("Standards", 0.127669)
        ], 0.347042),
        n("Moral", 0.051058, [
          n("Guidelines", 0.447876),
          n("Standards", 0.211562),
          n("Framework", 0.145404),
          n("Code", 0.077829)
        ])
      ]),
      n("Artificial", 0.041751, [
        n("Intelligence", 0.989503, [
          n("Ethics", 0.162102),
          n("Rules", 0.111411),
          n("Guidelines", 0.09832)
        ], 0.267262),
        n("Ethics", 0.006667, [
          n("Framework", 0.394515),
          n("Standards", 0.239285),
          n("Guidelines", 0.211169),
          n("Pledge", 0.01578)
        ]),
        n("Intellect", 0.001479, [
          n("Guidelines", 0.232622),
          n("Ethics", 0.232622),
          n("Standards", 0.124513),
          n("Regulations", 0.085577)
        ]),
        n("Intelligences", 0.00017)
      ]),
      n("New", 0.001619, [
        n("AI", 0.995921, [
          n("Guidelines", 0.317036),
          n("Regulations", 0.246908),
          n("Standards", 0.116631),
          n("Ethics", 0.090832, [
            n("Rules", 0.024137),
            n("Guidelines", 0.01292),
            n("Pact", 0.011402)
          ], 0.905751)
        ]),
        n("Ethics", 0.001497, [
          n("Guidelines", 0.368972),
          n("Framework", 0.25359),
          n("Standards", 0.15381),
          n("Rules", 0.044067)
        ]),
        n("Tech", 0.0008, [
          n("Guidelines", 0.283334),
          n("Regulations", 0.220661),
          n("Standards", 0.220661),
          n("Ethics", 0.133838, [
            n("Rules", 0.00049),
            n("Guidelines", 0.00038)
          ], 0.997439)
        ]),
        n("Artificial", 0.0008, [
          n("Guidelines", 0.296102),
          n("Intelligence", 0.296102, [
            n("Rules", 0.008245),
            n("Laws", 0.006836)
          ], 0.963609),
          n("Ethics", 0.230604, [
            n("Guidelines", 0.00058)
          ], 0.996977),
          n("Standards", 0.066069)
        ])
      ]),
      n("Robotic", 0.001119, [
        n("Ethics", 0.316376, [
          n("Framework", 0.28694),
          n("Standards", 0.253224),
          n("Guidelines", 0.223469)
        ], 0.041338),
        n("Responsibility", 0.149445, [
          n("Guidelines", 0.195738),
          n("Standards", 0.172738),
          n("Framework", 0.076652)
        ], 0.322717),
        n("AI", 0.149445, [
          n("Ethics", 0.272697),
          n("Guidelines", 0.187422),
          n("Standards", 0.165399),
          n("Regulations", 0.113677)
        ]),
        n("Accountability", 0.131885, [
          n("Standards", 0.299929),
          n("Guidelines", 0.097373),
          n("Framework", 0.062868)
        ], 0.339864)
      ])
    ]),
    n("Behind", 0.003971, [
      n("AI", 0.842854, [
        n("Ethics", 0.787382, [
          n("Framework", 0.58636),
          n("Guidelines", 0.148255),
          n("Standards", 0.101894),
          n("Pledge", 0.037335)
        ]),
        n("Ethical", 0.041721, [
          n("Framework", 0.427421),
          n("Guidelines", 0.293762),
          n("Standards", 0.228782),
          n("Code", 0.012907)
        ]),
        n("Morality", 0.039147, [
          n("Guidelines", 0.259095),
          n("Standards", 0.22865),
          n("Framework", 0.201783),
          n("Pledge", 0.095316)
        ]),
        n("Moral", 0.026943, [
          n("Guidelines", 0.428216),
          n("Framework", 0.202275),
          n("Code", 0.157532),
          n("Standards", 0.139021)
        ])
      ]),
      n("New", 0.061056, [
        n("AI", 0.994476, [
          n("Guidelines", 0.311039),
          n("Regulations", 0.188655),
          n("Standards", 0.114425),
          n("Ethics", 0.114425, [
            n("Pact", 0.023381),
            n("Rules", 0.020634),
            n("Guidelines", 0.014182)
          ], 0.877381)
        ]),
        n("Ethics", 0.002175, [
          n("Framework", 0.405405),
          n("Guidelines", 0.278631),
          n("Standards", 0.102502),
          n("Pledge", 0.048179)
        ]),
        n("Artificial", 0.00192, [
          n("Intelligence", 0.377925, [
            n("Guidelines", 0.023091),
            n("Rules", 0.020378),
            n("Code", 0.014909)
          ], 0.866485),
          n("Ethics", 0.294328, [
            n("Guidelines", 0.00051),
            n("Law", 0.00043)
          ], 0.996269),
          n("Guidelines", 0.178519),
          n("Standards", 0.02416)
        ]),
        n("Tech", 0.00055, [
          n("Guidelines", 0.290154),
          n("Standards", 0.175987),
          n("Regulations", 0.155308),
          n("Ethics", 0.155308, [
            n("Pact", 0.00085),
            n("Rules", 0.00066)
          ], 0.995181)
        ])
      ]),
      n("Artificial", 0.032681, [
        n("Intelligence", 0.975681, [
          n("Guidelines", 0.161118),
          n("Code", 0.161118),
          n("Standards", 0.142186),
          n("Ethics", 0.142186, [
            n("Pact", 0.00015)
          ], 0.999083)
        ]),
        n("Ethics", 0.01787, [
          n("Framework", 0.546367),
          n("Standards", 0.156537),
          n("Guidelines", 0.138143),
          n("Accord", 0.024006)
        ]),
        n("AI", 0.001662, [
          n("Guidelines", 0.262405),
          n("Ethics", 0.204361, [
            n("Guidelines", 0.008693),
            n("Framework", 0.008166),
            n("Pact", 0.00677)
          ], 0.943857),
          n("Standards", 0.109387),
          n("Regulations", 0.109387)
        ]),
        n("Intellect", 0.001383, [
          n("Guidelines", 0.228348),
          n("Ethics", 0.177837),
          n("Standards", 0.177837),
          n("Code", 0.084004)
        ])
      ]),
      n("Robotic", 0.004612, [
        n("Ethics", 0.430219, [
          n("Framework", 0.261701),
          n("Standards", 0.23095),
          n("Guidelines", 0.15873),
          n("Pledge", 0.066168)
        ]),
        n("AI", 0.139672, [
          n("Guidelines", 0.234942),
          n("Ethics", 0.234942),
          n("Standards", 0.207336),
          n("Regulations", 0.076275)
        ]),
        n("Responsibility", 0.095995, [
          n("Guidelines", 0.151891),
          n("Standards", 0.134043),
          n("Pledge", 0.092126)
        ], 0.134043),
        n("Accountability", 0.065976, [
          n("Standards", 0.254985),
          n("Framework", 0.154656),
          n("Pledge", 0.120446),
          n("Guidelines", 0.093804)
        ])
      ])
    ])
  ]),
  // ════════════════════════════════════════════════════════════
  // BRANCH 3: Agrees (0.185983)
  // ════════════════════════════════════════════════════════════
  n("Agrees", 0.185983, [
    n("on", 0.85016, [
      n("AI", 0.923934, [
        n("Ethics", 0.56671, [
          n("Framework", 0.71455),
          n("Standards", 0.12417),
          n("Guidelines", 0.10958),
          n("Regulations", 0.01155)
        ]),
        n("Regulations", 0.086908, [
          n("Framework", 0.434084),
          n("Finally", 0.019072),
          n("Strictly", 0.014661)
        ], 0.407784),
        n("Morality", 0.052524, [
          n("Standards", 0.300885),
          n("Guidelines", 0.23433),
          n("Framework", 0.206795),
          n("Rules", 0.059248)
        ]),
        n("Governance", 0.052712, [
          n("Framework", 0.179545),
          n("Rules", 0.1089),
          n("Standards", 0.084811)
        ], 0.488054)
      ]),
      n("Artificial", 0.021729, [
        n("Intelligence", 0.868372, [
          n("Rules", 0.157362),
          n("Code", 0.157362),
          n("Standards", 0.138872),
          n("Ethics", 0.122554)
        ]),
        n("Ethics", 0.080771, [
          n("Framework", 0.645096),
          n("Standards", 0.163106),
          n("Guidelines", 0.087304)
        ], 0.017191),
        n("AI", 0.010931, [
          n("Standards", 0.280798),
          n("Guidelines", 0.218686),
          n("Regulations", 0.170313),
          n("Ethics", 0.091162, [
            n("Guidelines", 0.002436),
            n("Framework", 0.001782)
          ], 0.987957)
        ]),
        n("Guidelines", 0.007513, [
          n("Strictly", 0.005558),
          n("Framework", 0.005801)
        ], 0.95752)
      ]),
      n("Unified", 0.014934, [
        n("AI", 0.988958, [
          n("Standards", 0.305022),
          n("Ethics", 0.209638, [
            n("Rules", 0.0008),
            n("Guidelines", 0.00075)
          ], 0.995511),
          n("Guidelines", 0.163266),
          n("Regulations", 0.099026)
        ]),
        n("Ethics", 0.005881, [
          n("Framework", 0.304113),
          n("Guidelines", 0.236844),
          n("Standards", 0.111877)
        ], 0.16278),
        n("Artificial", 0.004042, [
          n("Intelligence", 0.583079, [
            n("Rules", 0.039001),
            n("Code", 0.030374),
            n("Standards", 0.030374)
          ], 0.783364),
          n("Ethics", 0.189298),
          n("Guidelines", 0.101324)
        ], 0.032895),
        n("Ethical", 0.00037, [
          n("Guidelines", 0.383675),
          n("AI", 0.298806),
          n("Framework", 0.159939),
          n("Standards", 0.124561)
        ])
      ]),
      n("Robotic", 0.009601, [
        n("Ethics", 0.394019, [
          n("Framework", 0.289956),
          n("Standards", 0.155202),
          n("Guidelines", 0.136966)
        ], 0.199284),
        n("AI", 0.270805, [
          n("Standards", 0.327178),
          n("Ethics", 0.224866),
          n("Guidelines", 0.175126),
          n("Regulations", 0.082724)
        ]),
        n("Accountability", 0.06847, [
          n("Standards", 0.411665),
          n("Framework", 0.151443),
          n("Guidelines", 0.081062)
        ], 0.194457),
        n("Guidelines", 0.041529, [
          n("Framework", 0.100942),
          n("Strict", 0.039529)
        ], 0.655476)
      ])
    ]),
    n("AI", 0.076759, [
      n("Ethics", 0.725723, [
        n("Framework", 0.84821, [
          n("Guidelines", 0.310566),
          n("Standards", 0.061154),
          n("Globally", 0.050698)
        ], 0.129463),
        n("Guidelines", 0.04223, [
          n("Framework", 0.462995),
          n("Finally", 0.066701),
          n("Strictly", 0.061764),
          n("Nationwide", 0.051947)
        ]),
        n("Standards", 0.017604, [
          n("Framework", 0.496126),
          n("Globally", 0.067143),
          n("Pact", 0.023204)
        ], 0.133531),
        n("Treaty", 0.011366, [
          n("Framework", 0.922521),
          n("Guidelines", 0.012362),
          n("Standards", 0.007498)
        ], 0.013159)
      ]),
      n("Regulations", 0.059571, [
        n("Framework", 0.706629, [
          n("Globally", 0.010285)
        ], 0.946137),
        n("Nationally", 0.039815),
        n("Nationwide", 0.033049, [
          n("Pact", 0.005664),
          n("Framework", 0.005321)
        ], 0.963086),
        n("Guidelines", 0.027399, [
          n("Nationwide", 0.003076),
          n("Strictly", 0.002005)
        ], 0.97797)
      ]),
      n("Regulation", 0.046394, [
        n("Framework", 0.850407, [
          n("Globally", 0.008839),
          n("Guidelines", 0.006884)
        ], 0.91915),
        n("Guidelines", 0.047977, [
          n("Nationwide", 0.013035),
          n("Finally", 0.007427),
          n("Strictly", 0.006768)
        ], 0.913844),
        n("Standards", 0.029099, [
          n("Globally", 0.01529),
          n("Nationwide", 0.003011)
        ], 0.958628),
        n("Milestone", 0.008329, [
          n("Reached", 0.547697),
          n("Achieved", 0.039675),
          n("Deal", 0.029027)
        ], 0.201487)
      ]),
      n("Governance", 0.046394, [
        n("Framework", 0.816287, [
          n("Guidelines", 0.016621),
          n("Finally", 0.007376)
        ], 0.878238),
        n("Standards", 0.075926, [
          n("Globally", 0.020272),
          n("Framework", 0.010851)
        ], 0.873524),
        n("Guidelines", 0.031651, [
          n("Strictly", 0.05323),
          n("Finally", 0.034803),
          n("Nationwide", 0.032694)
        ], 0.699038),
        n("Principles", 0.019197, [
          n("Globally", 0.03051),
          n("Strictly", 0.010394),
          n("Finally", 0.009305)
        ], 0.837604)
      ])
    ]),
    n("to", 0.036259, [
      n("AI", 0.679967, [
        n("Ethics", 0.514451, [
          n("Framework", 0.699596),
          n("Standards", 0.121572),
          n("Guidelines", 0.107286),
          n("Regulations", 0.023939)
        ]),
        n("Regulations", 0.078894, [
          n("Framework", 0.637084),
          n("Strictly", 0.029355),
          n("Guidelines", 0.020479)
        ], 0.182528),
        n("Regulation", 0.078894, [
          n("Framework", 0.776138),
          n("Standards", 0.105039),
          n("Guidelines", 0.056223)
        ], 0.007609),
        n("Morality", 0.061076, [
          n("Standards", 0.480911),
          n("Framework", 0.227166),
          n("Guidelines", 0.176917),
          n("Pledge", 0.016456)
        ])
      ]),
      n("Unified", 0.104276, [
        n("AI", 0.986094, [
          n("Standards", 0.321951),
          n("Guidelines", 0.172328),
          n("Ethics", 0.152079, [
            n("Guidelines", 0.003129),
            n("Rules", 0.00294),
            n("Standards", 0.002437)
          ], 0.983248),
          n("Regulations", 0.118439)
        ]),
        n("Artificial", 0.007529, [
          n("Intelligence", 0.640099, [
            n("Code", 0.059245),
            n("Rules", 0.040718),
            n("Standards", 0.035934)
          ], 0.721749),
          n("Ethics", 0.126043, [
            n("Guidelines", 0.00075),
            n("Framework", 0.00066)
          ], 0.995892),
          n("Guidelines", 0.111233),
          n("Standards", 0.028124)
        ]),
        n("Ethics", 0.005175, [
          n("Framework", 0.493044),
          n("Guidelines", 0.232897),
          n("Standards", 0.181381),
          n("Code", 0.019117)
        ]),
        n("Ethical", 0.00041, [
          n("Guidelines", 0.334818),
          n("AI", 0.295476),
          n("Standards", 0.179215),
          n("Framework", 0.158157)
        ])
      ]),
      n("Robust", 0.041096, [
        n("AI", 0.990328, [
          n("Regulations", 0.288229),
          n("Standards", 0.254361),
          n("Guidelines", 0.17482),
          n("Governance", 0.082579)
        ]),
        n("Artificial", 0.003152, [
          n("Intelligence", 0.410242),
          n("Ethics", 0.193785),
          n("Guidelines", 0.193785),
          n("Standards", 0.05552)
        ]),
        n("Ethics", 0.002455, [
          n("Framework", 0.465224),
          n("Guidelines", 0.219756),
          n("Standards", 0.171146),
          n("Rules", 0.033701)
        ]),
        n("Regulations", 0.001489, [
          n("Guidelines", 0.00026)
        ], 0.996755)
      ]),
      n("Strict", 0.026365, [
        n("AI", 0.997291, [
          n("Regulations", 0.636596),
          n("Guidelines", 0.160957),
          n("Governance", 0.059213),
          n("Standards", 0.052255)
        ]),
        n("Artificial", 0.001925, [
          n("Intelligence", 0.46291, [
            n("Laws", 0.171492),
            n("Rules", 0.151342),
            n("Guidelines", 0.04336)
          ], 0.528234),
          n("Guidelines", 0.318153),
          n("Ethics", 0.062648, [
            n("Laws", 0.00095),
            n("Guidelines", 0.00058)
          ], 0.996232),
          n("Standards", 0.029593)
        ]),
        n("New", 0.00034, [
          n("Regulations", 0.306995),
          n("Guidelines", 0.270922),
          n("AI", 0.270922, [
            n("Rules", 0.45435),
            n("Laws", 0.31227),
            n("Guidelines", 0.06149),
            n("Standards", 0.042261)
          ]),
          n("Standards", 0.041547)
        ]),
        n("Tech", 0.00018, [
          n("Regulations", 0.640773),
          n("Governance", 0.098266),
          n("Guidelines", 0.076529),
          n("Regulation", 0.046417)
        ])
      ])
    ])
  ]),
  // ════════════════════════════════════════════════════════════
  // BRANCH 4: Achieves (0.013489)
  // ════════════════════════════════════════════════════════════
  n("Achieves", 0.013489, [
    n("Landmark", 0.473354, [
      n("AI", 0.99176, [
        n("Ethics", 0.856543, [
          n("Accord", 0.660818),
          n("Deal", 0.147448),
          n("Agreement", 0.06965),
          n("Consensus", 0.061466)
        ]),
        n("Accord", 0.033212, [
          n("Agreement", 0.120644),
          n("Finally", 0.036794),
          n("Deal", 0.028655)
        ], 0.61268),
        n("Ethical", 0.010755),
        n("Agreement", 0.013845, [
          n("Framework", 0.084489),
          n("Deal", 0.0658),
          n("Finally", 0.061813)
        ], 0.403074)
      ]),
      n("Artificial", 0.007572, [
        n("Intelligence", 0.767138, [
          n("Accord", 0.818457),
          n("Pact", 0.076128),
          n("Deal", 0.046174),
          n("Agreement", 0.031735)
        ]),
        n("Agreement", 0.171172, [
          n("Settlement", 0.01104),
          n("Deal", 0.010372)
        ], 0.944686),
        n("Ethics", 0.023166, [
          n("Accord", 0.617828),
          n("Deal", 0.227286),
          n("Pact", 0.065119),
          n("Agreement", 0.039496)
        ]),
        n("AI", 0.020444, [
          n("Accord", 0.827434),
          n("Agreement", 0.052896),
          n("Consensus", 0.024986),
          n("Accordance", 0.015155)
        ])
      ]),
      n("Agreement", 0.00038, [
        n("on", 0.96722, [
          n("AI", 0.908653),
          n("Ethics", 0.084518),
          n("Artificial", 0.002252),
          n("A", 0.00094)
        ]),
        n("in", 0.003617, [
          n("AI", 0.645305),
          n("Ethics", 0.346653),
          n("Artificial", 0.002054)
        ]),
        n("with", 0.003398, [
          n("AI", 0.906667),
          n("Tech", 0.031024),
          n("Industry", 0.006922),
          n("Reg", 0.006503)
        ])
      ])
    ]),
    n("AI", 0.25337, [
      n("Ethics", 0.905337, [
        n("Consensus", 0.260955, [
          n("Agreement", 0.404332),
          n("Framework", 0.261056),
          n("Finally", 0.108825),
          n("Deal", 0.03319)
        ]),
        n("Framework", 0.230376, [
          n("Agreement", 0.343619),
          n("Consensus", 0.267374),
          n("Deal", 0.098448),
          n("Accord", 0.08688)
        ]),
        n("Breakthrough", 0.065999, [
          n("Agreement", 0.57188),
          n("Finally", 0.068301),
          n("Deal", 0.038917),
          n("Together", 0.036559)
        ]),
        n("Landmark", 0.058244, [
          n("Agreement", 0.401769),
          n("Accord", 0.35456),
          n("Deal", 0.167482),
          n("Pact", 0.025684)
        ])
      ]),
      n("Governance", 0.01879, [
        n("Breakthrough", 0.357083, [
          n("Agreement", 0.438213),
          n("Finally", 0.097778),
          n("Deal", 0.046187)
        ], 0.081061),
        n("Milestone", 0.277558, [
          n("Agreement", 0.458447),
          n("Finally", 0.216555),
          n("Together", 0.040059),
          n("Today", 0.020143)
        ]),
        n("Consensus", 0.079637, [
          n("Agreement", 0.049847),
          n("Finally", 0.032184)
        ], 0.844306),
        n("Agreement", 0.070314, [
          n("Finally", 0.016705),
          n("Framework", 0.006542)
        ], 0.918601)
      ]),
      n("Morality", 0.012878, [
        n("Framework", 0.246521, [
          n("Agreement", 0.396237),
          n("Consensus", 0.165176),
          n("Deal", 0.128639),
          n("Accord", 0.068856)
        ]),
        n("Consensus", 0.131906),
        n("Accord", 0.102765, [
          n("Finally", 0.092491),
          n("Agreement", 0.038556),
          n("Framework", 0.024894)
        ], 0.642015),
        n("Standards", 0.080034, [
          n("Deal", 0.254082),
          n("Agreement", 0.197879),
          n("Pact", 0.068385)
        ], 0.224227)
      ]),
      n("Regulation", 0.012914, [
        n("Breakthrough", 0.352598, [
          n("Finally", 0.171244),
          n("Agreement", 0.160869),
          n("Today", 0.049062)
        ], 0.249159),
        n("Milestone", 0.310398, [
          n("Finally", 0.319215),
          n("Agreement", 0.264639),
          n("Together", 0.05211),
          n("Today", 0.029692)
        ]),
        n("Consensus", 0.146832, [
          n("Finally", 0.082103),
          n("Agreement", 0.032152),
          n("Nationwide", 0.009212)
        ], 0.778972),
        n("Landmark", 0.037163, [
          n("Deal", 0.418174),
          n("Agreement", 0.174321),
          n("Accord", 0.082343)
        ], 0.253636)
      ])
    ]),
    n("Historic", 0.135619, [
      n("AI", 0.99318, [
        n("Ethics", 0.784988, [
          n("Accord", 0.545918),
          n("Consensus", 0.200826),
          n("Agreement", 0.083719),
          n("Deal", 0.057539)
        ]),
        n("Governance", 0.039082, [
          n("Accord", 0.373039),
          n("Deal", 0.373039),
          n("Pact", 0.094319),
          n("Agreement", 0.083236)
        ]),
        n("Morality", 0.026222, [
          n("Accord", 0.535255),
          n("Pact", 0.19691),
          n("Deal", 0.105398),
          n("Pledge", 0.049787)
        ]),
        n("Regulation", 0.020919, [
          n("Deal", 0.318689),
          n("Accord", 0.248195),
          n("Agreement", 0.193295),
          n("Pact", 0.062754)
        ])
      ]),
      n("Artificial", 0.005906, [
        n("Intelligence", 0.592241, [
          n("Accord", 0.832024),
          n("Pact", 0.05319),
          n("Agreement", 0.036557),
          n("Deal", 0.022173)
        ]),
        n("Agreement", 0.317004, [
          n("Framework", 0.163762),
          n("Settlement", 0.099327),
          n("Deal", 0.049945)
        ], 0.418182),
        n("Consensus", 0.033399, [
          n("Agreement", 0.505343),
          n("Deal", 0.154121),
          n("Framework", 0.077497)
        ], 0.12003),
        n("Ethics", 0.015783, [
          n("Accord", 0.760944),
          n("Deal", 0.080203),
          n("Agreement", 0.055123),
          n("Pact", 0.033434)
        ])
      ]),
      n("Agreement", 0.00038, [
        n("on", 0.991501, [
          n("AI", 0.91547, [
            n("Ethics", 0.252958),
            n("Governance", 0.034234),
            n("Regulations", 0.016171)
          ], 0.606814),
          n("Ethics", 0.075146, [
            n("Guidelines", 0.001764)
          ], 0.994273),
          n("Artificial", 0.003302, [
            n("Intelligence", 0.379367)
          ], 0.615776),
          n("A.I", 0.00056)
        ]),
        n("Over", 0.002401, [
          n("AI", 0.974944, [
            n("Ethics", 0.031493),
            n("Governance", 0.00748)
          ], 0.927846),
          n("Ethics", 0.012273, [
            n("Guidelines", 0.00062)
          ], 0.997262),
          n("Artificial", 0.001882, [
            n("Intelligence", 0.297179)
          ], 0.699189),
          n("Algorithms", 0.001142)
        ]),
        n("in", 0.00187, [
          n("AI", 0.767032, [
            n("Ethics", 0.485238),
            n("Governance", 0.074414),
            n("Regulation", 0.03102)
          ], 0.333499),
          n("Ethics", 0.219758, [
            n("Guidelines", 0.00058)
          ], 0.996315),
          n("Artificial", 0.004025, [
            n("Intelligence", 0.437659)
          ], 0.555149),
          n("Ethic", 0.00063)
        ])
      ]),
      n("Consensus", 7.4e-05, [
        n("on", 0.997064, [
          n("AI", 0.912738),
          n("Ethics", 0.074922),
          n("Artificial", 0.002262),
          n("Regulation", 0.001762)
        ]),
        n("in", 0.0009, [
          n("AI", 0.876717),
          n("Ethics", 0.104709),
          n("Artificial", 0.006694),
          n("Tech", 0.00159)
        ]),
        n("Over", 0.00079, [
          n("AI", 0.978166),
          n("Ethics", 0.010866),
          n("Algorithms", 0.001565),
          n("Artificial", 0.001382)
        ])
      ])
    ]),
    n("Groundbreaking", 0.043993, [
      n("AI", 0.989516, [
        n("Ethics", 0.807153, [
          n("Accord", 0.619451),
          n("Consensus", 0.121977),
          n("Deal", 0.083834),
          n("Agreement", 0.044873)
        ]),
        n("Accord", 0.045536, [
          n("Agreement", 0.058217),
          n("Finally", 0.029273),
          n("Settlement", 0.01668)
        ], 0.754973),
        n("Morality", 0.031037),
        n("Ethical", 0.014263)
      ]),
      n("Artificial", 0.009701, [
        n("Intelligence", 0.70217, [
          n("Accord", 0.772977),
          n("Pact", 0.11854),
          n("Deal", 0.029972),
          n("Agreement", 0.020599)
        ]),
        n("Agreement", 0.201175, [
          n("Framework", 0.005211),
          n("Settlement", 0.003812)
        ], 0.971312),
        n("Ethics", 0.050865, [
          n("Accord", 0.688605),
          n("Deal", 0.105601),
          n("Pact", 0.093193),
          n("Agreement", 0.023563)
        ]),
        n("Consensus", 0.0165)
      ]),
      n("Tech", 0.00012, [
        n("Ethics", 0.612533, [
          n("Accord", 0.63695),
          n("Deal", 0.161046),
          n("Pact", 0.097679),
          n("Agreement", 0.024697)
        ]),
        n("Accord", 0.120615, [
          n("Agreement", 0.013668),
          n("Settlement", 0.01)
        ], 0.919503),
        n("Morality", 0.052097),
        n("Harmony", 0.030496, [
          n("Agreement", 0.230572),
          n("Accord", 0.230572),
          n("Deal", 0.17957),
          n("Pact", 0.148869)
        ])
      ]),
      n("Agreement", 0.00012, [
        n("on", 0.962342, [
          n("AI", 0.887093),
          n("Ethics", 0.105948),
          n("Artificial", 0.002492),
          n("A", 0.00071)
        ]),
        n("with", 0.002949, [
          n("AI", 0.930555),
          n("Tech", 0.019313),
          n("Artificial", 0.007563),
          n("Reg", 0.00627)
        ]),
        n("Framework", 0.002602)
      ])
    ])
  ])
]);
