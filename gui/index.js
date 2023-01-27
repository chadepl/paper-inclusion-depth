


// FUNCTIONS TODO: move to ES6 module

async function post_json(url, data={}){
        let response = await fetch(url, {
            method: "POST",
            headers: {
                "Accept": "application/json",
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });
        let response_json = await response.json();
        return response_json;
    }

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

function render_image(num_rows, num_cols, ctx, array, cmap, reversed = false, zero_transparency = false){
    const min = d3.min(array);
    const max = d3.max(array);
    const interpolator = d3.scaleSequential().interpolator(cmap);
    if (reversed) {
        interpolator.domain([max, min]);
    } else {
        interpolator.domain([min, max]);
    }

    const image_data = ctx.createImageData(num_cols, num_rows);
    const data = image_data.data;
    let i, val;
    for(let r=0; r<num_rows; r++){
        for(let c=0; c<num_cols; c++){
            val = d3.color(interpolator(array[[r * num_rows + c]]));
            i = 4 * ((r * num_cols) + c);
            data[i + 0] = val.r;  // red
            data[i + 1] = val.g;  // green
            data[i + 2] = val.b;  // blue
            data[i + 3] = val.opacity * 255;  // alpha
        }
    }
    ctx.putImageData(image_data, 0, 0);
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

// FUNCTIONS (network/async)
async function fetch_available_datasets(server_url){
    const response = await fetch(`${server_url}/available_datasets`);
    const response_json = await response.json();
    let available_datasets = new Map();
    response_json.available_datasets.forEach(ad => available_datasets.set(ad.name, ad));
    return available_datasets;
}

async function fetch_ensemble_data(server_url, dataset_details){
    const endpoint = dataset_details.endpoint;
    const kwargs = dataset_details.kwargs;
    if(endpoint.includes("ellipses_dataset")){
        let data = {
            num_rows: state.menu.d_ellipses.grid_size.value,
            num_cols: state.menu.d_ellipses.grid_size.value,
            num_members: state.menu.d_ellipses.num_members.value,
            kwargs: kwargs
        };
        const response = await post_json(`${server_url}${endpoint}`, data);
        return response;
    }else {
        console.log("[fetch_ensemble_data] Selected dataset is currently not supported.")
    }
}

////////
// UI //
////////

// Application state

let state = {};

state.server_url = "http://localhost:6969";
state.ensemble_data = null;

// - UI-related

state.menu = {};
state.menu.general = {};
state.menu.general.available_ensemble_vis = new Map();
state.menu.general.available_ensemble_vis.set("member_plot", {name: "Member plot"});
state.menu.general.available_ensemble_vis.set("spaghetti_plot", {name: "Spaghetti plot"});
state.menu.general.available_ensemble_vis.set("mask_overlay", {name: "Mask overlay"});
state.menu.general.available_ensemble_vis.set("contour_boxplot", {name: "Contour boxplot"});
state.menu.general.vis_method = "member_plot";

state.menu.general.ensemble_vis = "spaghetti_plot";
state.menu.general.available_datasets = new Map();  // To fetch later
state.menu.general.dataset = null;  // To set later when datasets are available

state.menu.d_ellipses = {};
state.menu.d_ellipses.grid_size = {min: 200, max: 800, value:500, step: 100};  // value = num_cols (w) = num_rows (h)
state.menu.d_ellipses.num_members = {min: 1, max: 100, value: 10, step: 1};

state.menu.vis_member_plot = {};
state.menu.vis_member_plot.member = 0;

// UI elements

// - Views

let viewer_element = document.getElementById("viewer");
let scatter_element = document.getElementById("scatter");
let depth_explorer_element = document.getElementById("depth-explorer");

// - Menu

let menu_el = {};  // object with elements in the menu for easy access

menu_el.general = {};
menu_el.general.select_available_datasets = document.getElementById("select_available_datasets");
menu_el.general.select_available_vis_methods = document.getElementById("ensemble_vis_method");
state.menu.general.available_ensemble_vis.forEach((d_v, d_k) => menu_el.general.select_available_vis_methods.add(new Option(d_v.name, d_k)));

menu_el.d_ellipses = {}
menu_el.d_ellipses.container = null;  // to be able to hide/reveal the menu on demand
menu_el.d_ellipses.range_grid_size = document.getElementById("range_size");
menu_el.d_ellipses.range_grid_size.min = state.menu.d_ellipses.grid_size.min;
menu_el.d_ellipses.range_grid_size.max = state.menu.d_ellipses.grid_size.max;
menu_el.d_ellipses.range_grid_size.step = state.menu.d_ellipses.grid_size.step;
menu_el.d_ellipses.range_grid_size.value = state.menu.d_ellipses.grid_size.value;
document.getElementById("range_size-min").textContent = state.menu.d_ellipses.grid_size.min;
document.getElementById("range_size-max").textContent = state.menu.d_ellipses.grid_size.max;
document.getElementById("range_size-value").textContent = state.menu.d_ellipses.grid_size.value;
menu_el.d_ellipses.range_grid_size.addEventListener("change", (event) => {
    state.menu.d_ellipses.grid_size.value = parseInt(menu_el.d_ellipses.range_grid_size.value);
    configure_app_for_dataset();
});

menu_el.d_ellipses.range_num_members = document.getElementById("range_num_members");
menu_el.d_ellipses.range_num_members.min = state.menu.d_ellipses.num_members.min;
menu_el.d_ellipses.range_num_members.max = state.menu.d_ellipses.num_members.max;
menu_el.d_ellipses.range_num_members.step = state.menu.d_ellipses.num_members.step;
menu_el.d_ellipses.range_num_members.value = state.menu.d_ellipses.num_members.value;
document.getElementById("range_num_members-min").textContent = state.menu.d_ellipses.num_members.min;
document.getElementById("range_num_members-max").textContent = state.menu.d_ellipses.num_members.max;
menu_el.d_ellipses.range_num_members.addEventListener("change", (event) => {
    state.menu.d_ellipses.num_members.value = menu_el.d_ellipses.range_num_members.value;
    document.getElementById("range_num_members-value").textContent = state.menu.d_ellipses.num_members.value;
    configure_app_for_dataset();
});

menu_el.vis_member_plot = {}
menu_el.vis_member_plot.container = document.getElementById("menu_member_plot");
menu_el.vis_member_plot.range_mp_member_id = document.getElementById("range_mp_member_id");
menu_el.vis_member_plot.range_mp_member_id.min = 0;
menu_el.vis_member_plot.range_mp_member_id.max = 10;
menu_el.vis_member_plot.range_mp_member_id.value = state.menu.vis_member_plot.member;
menu_el.vis_member_plot.range_mp_member_id.addEventListener("input", (event) => {
    state.menu.vis_member_plot.member = menu_el.vis_member_plot.range_mp_member_id.value;
    update_canvas();
})

menu_el.vis_spaghetti_plot = {}
menu_el.vis_spaghetti_plot.container = document.getElementById("menu_spaghetti_plot");

menu_el.vis_mask_overlay = {}
menu_el.vis_mask_overlay.container = document.getElementById("menu_mask_overlay");

menu_el.vis_contour_boxplot = {}
menu_el.vis_contour_boxplot.container = document.getElementById("menu_contour_boxplot");


// Event listeners

// - General
menu_el.general.select_available_datasets.addEventListener("change", (event) => onchange_select_available_datasets());
menu_el.general.select_available_vis_methods.addEventListener("change", (event) => onchange_select_available_vis_methods());


function onchange_select_available_datasets(){
    state.menu.general.dataset = menu_el.general.select_available_datasets.value;
    configure_app_for_dataset();
}

function onchange_select_available_vis_methods(){
    state.menu.general.vis_method = menu_el.general.select_available_vis_methods.value;
    update_menu();
    update_canvas();
}

// Execution chain
// load_available_datasets -> configure_app_for_dataset ->

// TODO: make it modular to be able to update individual portions
async function load_available_datasets() {
    state.menu.general.available_datasets = await fetch_available_datasets(state.server_url);
    state.menu.general.available_datasets.forEach((d_v, d_k) => menu_el.general.select_available_datasets.add(new Option(d_v.name, d_k)));
    state.menu.general.dataset = state.menu.general.available_datasets.keys().next().value;

    menu_el.general.select_available_datasets.dispatchEvent(new Event("change"));
}

load_available_datasets();

async function configure_app_for_dataset(){
    console.log("[configure_app_for_dataset]");
    const server_url = state.server_url;

    // Loading data
    let ensemble_data = await fetch_ensemble_data(server_url, state.menu.general.available_datasets.get(state.menu.general.dataset));
    state.ensemble_data = await ensemble_data;

    const depth_data_response = (await post_json(`${server_url}/ensemble_depths`, {
        num_rows: state.ensemble_data.num_rows,
        num_cols: state.ensemble_data.num_cols,
        ensemble: state.ensemble_data.ensemble.members.map(m => m.data)
    }));

    state.ensemble_data.ensemble_depths = await depth_data_response.depth_data;

    const path_promises = state.ensemble_data.ensemble.members.map(async function(m) {
        const path = (await post_json(`${server_url}/member_path_representation`, {
            num_cols: state.ensemble_data.num_cols,
            num_rows: state.ensemble_data.num_rows,
            isovalue: 0.5,
            array: m.data
        })).path;
        return path;
    });

    const paths = await Promise.all(path_promises);

    console.log(paths);

    state.ensemble_data.ensemble.members = state.ensemble_data.ensemble.members.map((m,i) => {
        m.path = paths[i];
        return m;
    });

    console.log(state.ensemble_data.ensemble.members);
    console.log(state);

    // const arrpro = state.ensemble_data.ensemble.members.map(async m => {
    //     return await post_json(`${server_url}/member_path_representation`, {
    //         num_cols: state.ensemble_data.num_cols,
    //         num_rows: state.ensemble_data.num_rows,
    //         isovalue: 0.5,
    //         array: m.data
    //     });
    // });
    //
    // console.log(arrpro);
    // const fulfilled = Promise.all(arrpro);
    // fulfilled.then(values => console.log(values));

    // Menu related stuff
    // - Configure dataset-specific menu
    menu_el.general.select_available_vis_methods.dispatchEvent(new Event("change"));
}


function update_menu(){
    // Hide all extra menus
    let menus = document.getElementsByClassName("vis_method_menu")
    for(let menu of menus){
        menu.style.display = "none";
    };
    if(state.menu.general.vis_method === "member_plot"){
        menu_el.vis_member_plot.container.style.display = "block";
    }else if (state.menu.general.vis_method === "spaghetti_plot"){
        menu_el.vis_spaghetti_plot.container.style.display = "block";
    }else if (state.menu.general.vis_method === "mask_overlay"){
        menu_el.vis_mask_overlay.container.style.display = "block";
    }else if (state.menu.general.vis_method === "contour_boxplot"){
        menu_el.vis_contour_boxplot.container.style.display = "block";
    }
}


let viewer_size = [0,0];
let scatter_size = [0,0];
function update_dimensions(){
    viewer_size = [viewer_element.offsetWidth, viewer_element.offsetHeight];
    scatter_size = [scatter_element.offsetWidth, scatter_element.offsetHeight];
    // TODO: update panels and canvas
}

update_dimensions();
//
// // Init viewer
let viewer_canvas = viewer_element.appendChild(document.createElement("canvas"));
viewer_canvas.setAttribute("id", "viewer_canvas")
viewer_canvas.height = Math.floor(Math.min(...viewer_size)); // canvas size is independent from raster size
viewer_canvas.width = Math.floor(Math.min(...viewer_size)); // canvas size is independent from raster size

console.log(viewer_canvas.height);

viewer_ctx = viewer_canvas.getContext("2d");


async function update_canvas(){

    // General variables

    const ensemble_data = state.ensemble_data;
    const num_members = ensemble_data.ensemble.length;
    const num_rows = ensemble_data.num_rows;
    const num_cols = ensemble_data.num_cols;

    const fields = ensemble_data.fields;
    const ensemble = ensemble_data.ensemble.members;
    const members_data = ensemble.map(m => m.data);
    const members_paths = ensemble.map(m => m.path);
    const members_feat = ensemble.map(m => m.features);

    const select_available_vis_methods = menu_el.general.select_available_vis_methods;

    // Visualization-related

    // - Clean-up previous stuff
    // - Initialize main canvas


    // - Initialize main sgv

    // - Initialize offscreen canvas
    const off_canvas = new OffscreenCanvas(num_cols, num_rows);
    const off_ctx = off_canvas.getContext("2d");

    //////////////////////////////////////
    // SHAPE VISUALIZATION (MAIN VIEWER //
    //////////////////////////////////////



    if(select_available_vis_methods.value == "member_plot"){

        viewer_ctx.clearRect(0,0, viewer_canvas.width, viewer_canvas.height);

        const member_id = state.menu.vis_member_plot.member;
        off_ctx.strokeStyle = "red";
        const path = await members_paths[member_id];

        off_ctx.beginPath();
        off_ctx.moveTo(path[0][1], path[0][0]);
        path.forEach(c => {
            off_ctx.lineTo(c[1], c[0]);
        });
        //render_image(num_rows, num_cols, off_ctx, members_data[member_id], d3.interpolateGreys);
        off_ctx.stroke();


    }else if(select_available_vis_methods.value == "mask_overlay"){
        // Vis method 1: mask overlay
        console.log("Ensemble vis method: mask overlay");
        let grid = (new Array(num_cols*num_rows)).fill(0.0);
        for(let m=0; m<num_members; m++){
            members_data[m].forEach((v, i) => grid[i] += v * (1/num_members));
        }

        render_image(num_rows, num_cols, off_ctx, grid, d3.interpolateGreys);

    } else if(select_available_vis_methods.value == "spaghetti_plot"){
        // Vis method 2: contour overlay
        console.log("Ensemble vis method: spaguetti plot");
        viewer_ctx.clearRect(0, 0, viewer_canvas.width, viewer_canvas.height);

        let member = members_data[0];

        //render_image(num_rows, num_cols, offctx, member, d3.interpolateGreys);

        const scale_color = d3.scaleSequential(d3.interpolateTurbo);

        for(let mid=0; mid<members_data.length; mid++){
            off_ctx.strokeStyle = scale_color(Math.random());
            //console.log(offctx.strokeStyle);
            let path = (await post_json(`${server_url}/member_path_representation`, {
                num_cols,
                num_rows,
                isovalue: 0.5,
                array: members_data[mid]
            })).path;
            off_ctx.beginPath();
            off_ctx.moveTo(path[0][1], path[0][0]);
            path.forEach(c => {
                off_ctx.lineTo(c[1], c[0]);
            });
            off_ctx.stroke();
        }


    } else if(select_available_vis_methods.value == "contour_boxplot") {

        // Vis method 3: contour boxplot
        console.log("Ensemble vis method: contour boxplot");
        // const depth_data = state.ensemble_data.ensemble_depths;
        // const depths = depth_data.depth_data;

        const num_subsets = ensemble_data.contour_boxplot.num_subsets;
        //const depth_data = ensemble_data.contour_boxplot.depth_data;
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

        const mask_median = members_data[median].data;

        render_image(grid_size, grid_size, off_ctx, mask_median, d3.interpolateGreys, reversed = false, zero_transparency = true);
        outliers.forEach(member_id => plot_isocurve(grid_size, grid_size, off_ctx, members_data[member_id].data, isolevel=0.5, line_params={width: 1, dash_pattern: [2,2]}));
        plot_isocurve(grid_size, grid_size, off_ctx, mask_median, isolevel=0.5, line_params={color: "teal"});

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

    const depth_data = state.ensemble_data.ensemble_depths;
    const depths = depth_data.depth_data;

    viewer_ctx.imageSmoothingEnabled = true;
    viewer_ctx.imageSmoothingQuality = "high";
    // viewer_ctx.drawImage(offcanvas, 0, 0);
    viewer_ctx.drawImage(off_canvas, 0, 0, viewer_canvas.width, viewer_canvas.height);
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
            values: members_feat
        },
        mark: 'point',
        encoding: {
            x: {field: "rotation_deg", type: 'quantitative', scale: {domain: d3.extent(members_feat, d => d.rotation_deg)}},
            y: {field: 'isovalue', type: 'quantitative', scale: {domain: d3.extent(members_feat, d => d.isovalue)}},
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
            values: depths
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

