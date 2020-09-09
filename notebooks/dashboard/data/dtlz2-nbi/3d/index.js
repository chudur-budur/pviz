
        var dtlz2_nbi_3d = {
            name: "DTLZ2_NBI",
            dim: 3,
            // f-values
            fvals: require("./dataf.json"),
            // tradeoff values
            mus: require("./mu.json"),
            // point sizes
            sizes: require("./sizes.json"),
            // knee indices
            kidx: require("./muid.json"),
            // centroid distances
            cdist: require("./cdist.json"),
            // color by centroid
            clr_c: require("./color-centroid.json"),
            // cv values
            cvs: null,
            // color by cv
            clr_cv: null,
        };
        
        export default dtlz2_nbi_3d;
    