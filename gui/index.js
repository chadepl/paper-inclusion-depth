


// FUNCTIONS TODO: move to ES6 module


function plot_isocurve(num_rows, num_cols, ctx, array, isolevel = 0.5, line_params = {}) {

    const w = num_rows;
    const h = num_cols;

    if (!("color" in line_params))
        line_params.color = "red";
    if (!("opacity" in line_params))
        line_params.opacity = 1.0;
    if (!("width" in line_params))
        line_params.width = 1.0;
    if (!("dash_pattern" in line_params))
        line_params.dash_pattern = [];

    // Plot contour as path
    const contour_extractor = d3.contours().size([h, w]);
    const contour_geo_path = contour_extractor.contour(array, isolevel);

    let color = d3.color(line_params.color);
    color.opacity = line_params.opacity;
    ctx.strokeStyle = color;
    ctx.lineWidth = line_params.width;
    ctx.setLineDash(line_params.dash_pattern);

    const path = d3.geoPath().context(ctx);
    ctx.beginPath();
    path(contour_geo_path);

    ctx.stroke();
    return line_params;
}


function render_image(num_rows, num_cols, ctx, array, cmap, reversed = false, zero_transparency = false) {
    // array is a 2D array with rows (h) in the first dim and cols (w) in the second
    // cmap is a d3 interpolator function that gets a number and returns a 4 number array representing an rgba color
    // TODO: see if there is a more efficient way to draw pixels in canvas
    const min = d3.min(array);
    const max = d3.max(array);
    const interpolator = d3.scaleSequential().interpolator(cmap);
    if (reversed) {
        interpolator.domain([max, min]);
    } else {
        interpolator.domain([min, max]);
    }
    let val;

    for (let i = 0; i <= array.length; i++){

    }

    for (let r = 0; r < num_rows; r++) {
        for (let c = 0; c < num_cols; c++) {
            val = d3.color(interpolator(array[[r * num_rows + c]]));
            if (zero_transparency && array[r][c] === min) {
                val.opacity = 0.0;
            }
            ctx.fillStyle = val;
            ctx.fillRect(c, r, 1, 1);
        }
    }
}


function render_binary_mask(ctx, array, color_fg="white", opacity_fg=1.0, color_bg="black", opacity_bg=0.0){
  const [nrows, ncols] = [ctx.canvas.height, ctx.canvas.width];
  let val, color;

  for(let r=0; r<nrows; r++){
    for(let c=0; c<ncols; c++){
      val = Math.round(array[r][c]);
      if(val == 1){
        color = d3.color(color_fg);
        color.opacity = opacity_fg;
      }else if(val == 0){
        color = d3.color(color_bg);
        color.opacity = opacity_bg;
      }
      ctx.fillStyle = color;
      ctx.fillRect(c, r, 1, 1);
    }
  }

}



// DECLARATIONS //

let grid_size = 300;
let num_members = 20;

let viewer_element = document.getElementById("viewer");
let scatter_element = document.getElementById("scatter");
let depth_explorer_element = document.getElementById("depth-explorer");

const range_size = document.getElementById("range_size");
const range_num_members = document.getElementById("range_num_members");

// - Initialize inputs
range_size.min = 200;
range_size.max = 800;
range_size.step = 100;
range_size.value = grid_size;
document.getElementById("range_size-min").textContent = range_size.min;
document.getElementById("range_size-max").textContent = range_size.max;

range_num_members.min = 1;
range_num_members.max = 100;
range_num_members.value = num_members;
document.getElementById("range_num_members-min").textContent = range_num_members.min;
document.getElementById("range_num_members-max").textContent = range_num_members.max;

// - Add listeners
range_size.addEventListener("input", update_menu);
range_num_members.addEventListener("input", update_menu);

function update_menu(){
    console.log("Should update menu");

    document.getElementById("range_size-value").textContent = range_size.value;
    document.getElementById("range_num_members-value").textContent = range_num_members.value;

}

update_menu();

let viewer_size = [0,0];
let scatter_size = [0,0];
function update_dimensions(){
    viewer_size = [viewer_element.offsetWidth, viewer_element.offsetHeight];
    scatter_size = [scatter_element.offsetWidth, scatter_element.offsetHeight];
}

update_dimensions();

// Init viewer
let viewer_canvas = viewer_element.appendChild(document.createElement("canvas"));
viewer_canvas.setAttribute("id", "viewer_canvas")
viewer_canvas.height = Math.floor(Math.min(...viewer_size)); // canvas size is independent from raster size
viewer_canvas.width = Math.floor(Math.min(...viewer_size)); // canvas size is independent from raster size

console.log(viewer_canvas.height);

viewer_ctx = viewer_canvas.getContext("2d");



state = {
    ensemble_data: {
        fields: {

        },
        masks: {}
    }
}

console.log(state);


async function fetch_ensemble_data(){
    grid_size = range_size.value;
    num_members = range_num_members.value;
    const response = await fetch(`http://localhost:6969/ensemble_data?size=${grid_size}&num_members=${num_members}`);
    const ensemble_data = await response.json();
    update_canvas(ensemble_data);
    //console.log(response);
    //console.log(ensemble_data);
}

async function update_canvas(ensemble_data){
    console.log(ensemble_data);

    const num_rows = ensemble_data.num_rows;
    const num_cols = ensemble_data.num_cols;
    const fields = ensemble_data.fields;
    const ensemble = ensemble_data.ensemble.members;
    const num_members = ensemble.length;

    /////////////////////////
    // SHAPE VISUALIZATION //
    /////////////////////////

    let offcanvas = new OffscreenCanvas(grid_size, grid_size);
    let offctx = offcanvas.getContext("2d");

    if(document.getElementById("ensemble_vis_method").value == "ev_mask_overlay"){
        // Vis method 1: mask overlay
        console.log("Ensemble vis method: mask overlay");
        let grid = (new Array(num_cols*num_rows)).fill(0.0);
        for(let m=0; m<num_members; m++){
            ensemble[m].data.forEach((v, i) => grid[i] += v * (1/num_members));
        }

        let scale_color = d3.scaleSequential(d3.extent(grid), d3.interpolateGreys);
        for(let r=0; r<num_rows; r++){
            for(let c=0; c<num_cols; c++){
                offctx.fillStyle = scale_color(grid[r*num_rows+c]);
                offctx.fillRect(c, r, 1, 1);
            }
        }
    } else if(document.getElementById("ensemble_vis_method").value == "ev_contour_overlay"){
        // Vis method 2: contour overlay
        console.log("Ensemble vis method: contour overlay");
        viewer_ctx.clearRect(0, 0, viewer_canvas.width, viewer_canvas.height);

    } else if(document.getElementById("ensemble_vis_method").value == "ev_contour_boxplot") {
        // Vis method 3: contour boxplot
        console.log("Ensemble vis method: contour boxplot");
        const num_subsets = ensemble_data.contour_boxplot.num_subsets;
        const depth_data = ensemble_data.contour_boxplot.depth_data;
        const dd_matrix = (new Array(num_members)).fill(null).map(d => (new Array(num_subsets)).fill(null).map(d => 0.0));
        const containment_epsilon = 0.1;
        depth_data.forEach(d => {
            dd_matrix[d.member_id][d.subset_id] = d.lc_frac <= containment_epsilon && d.rc_frac <= containment_epsilon ? 1.0 : 0.0;
        });

        const depths = [];
        dd_matrix.forEach(member_data => {
            depths.push(member_data.reduce((a, b) => a + b)/num_subsets);
        })

        depth_data.map(d => d.contained =  d.lc_frac <= containment_epsilon && d.rc_frac <= containment_epsilon ? 1.0 : 0.0);
        const depths_member_mean = Array.from(d3.rollup(depth_data, v => d3.mean(v, d => d.contained), d => d.member_id), ([key, value]) => ({id:key, value}));
        const median = depths_member_mean[d3.maxIndex(depths_member_mean, d => d.value)].id;  // median
        const hundred_percent = d3.sort(d3.filter(depths_member_mean, d => d.value !== 0.0),  (a,b) => d3.descending(a.value, b.value)).map(d => d.id)
        const fifty_percent = d3.filter(hundred_percent, (d,i) => i + 1 <= hundred_percent.length / 2);
        const outliers = d3.filter(depths_member_mean, d => d.value == 0).map(d => d.id);

        console.log(depths);
        console.log(depths_member_mean);
        console.log(median);
        console.log(hundred_percent);
        console.log(fifty_percent);
        console.log(outliers);

        const mask_median = ensemble[median].data;

        render_image(grid_size, grid_size, offctx, mask_median, d3.interpolateGreys, reversed = false, zero_transparency = true);
        outliers.forEach(member_id => plot_isocurve(grid_size, grid_size, offctx, ensemble[member_id].data, isolevel=0.5, line_params={width: 1, dash_pattern: [2,2]}));
        plot_isocurve(grid_size, grid_size, offctx, mask_median, isolevel=0.5, line_params={color: "teal"});

        const response = await fetch(`http://localhost:6969/contour_band`, {
            method: "POST",
            headers: {
                "Accept": "application/json",
                "Content-Type": "application/json"
            },
            body: JSON.stringify({a: 1, b: "test"})
        });
        const band = await response.json();
        console.log(band);

        viewer_ctx.clearRect(0, 0, viewer_canvas.width, viewer_canvas.height);

    }


    viewer_ctx.imageSmoothingEnabled = true;
    viewer_ctx.imageSmoothingQuality = "high";
    viewer_ctx.drawImage(offcanvas, 0, 0, viewer_canvas.width, viewer_canvas.height);
    //let image_data = offctx.getImageData(0, 0, viewer_size[0], viewer_size[1]);
    //console.log(image_data);

    ////////////////////////
    // SUPPORTING WIDGETS //
    ////////////////////////

    // Scatterplot

    let scatter_spec = {
        $schema: 'https://vega.github.io/schema/vega-lite/v5.json',
        description: 'A simple bar chart with embedded data.',
        width: "container",
        height: "container",
        data: {
            values: ensemble.map(d => d.features)
        },
        mark: 'point',
        encoding: {
            x: {field: "rotation_deg", type: 'quantitative', scale: {domain: d3.extent(ensemble, d => d.features.rotation_deg)}},
            y: {field: 'isovalue', type: 'quantitative', scale: {domain: d3.extent(ensemble, d => d.features.isovalue)}},
            size: {field: "ellipticity", type: "quantitative"}
        }
      };

    vegaEmbed("#scatter", scatter_spec);

    // Depth explorer

    let depth_explorer_spec = {
        $schema: 'https://vega.github.io/schema/vega-lite/v5.json',
        description: 'A simple bar chart with embedded data.',
        width: "container",
        height: "container",
        data: {
            values: ensemble_data.contour_boxplot.depth_data
        },
        mark: 'rect',
        encoding: {
            x: {field: "subset_id", type: 'ordinal'},
            y: {field: 'member_id', type: 'ordinal'},
            color: {field: "rc_frac", type: "quantitative"}
        }
    }

    vegaEmbed("#depth-explorer", depth_explorer_spec);

}


// let ensemble_data = await fetch("http://localhost:6969/ensemble_data", {method: "GET"});
// console.log(ensemble_data);

