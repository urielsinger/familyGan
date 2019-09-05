import os
import pickle as pkl

import numpy as np
from PIL import Image
from auto_tqdm import tqdm
from bokeh.layouts import column
from bokeh.layouts import row
from bokeh.models import CustomJS, ColumnDataSource, Slider
from bokeh.models import Image
from bokeh.models.glyphs import ImageURL
from bokeh.plotting import figure
from bokeh.plotting import show

from familyGan.load_data import get_files_from_path


def family_view_with_slider(pkl_folder_path):
    # TODO: not tested


    # Save folders in curr folder for bokeh access        
    os.makedirs("pics/", exist_ok=True)
    father_img_paths, mother_img_paths, child_img_paths = [], [], []
    for i, filep in enumerate(get_files_from_path(pkl_folder_path)):
        with open(filep, 'rb') as f:
            (father_image, father_latent_f), (mother_image, mother_latent_f), (child_image, child_latent_f) = pkl.load(
                f)

            father_img_p, mother_img_p, child_img_p = f'pics/{i}-F.png', f'pics/{i}-M.png', f'pics/{i}-C.png'

            father_image.save(father_img_p)
            mother_image.save(mother_img_p)
            child_image.save(child_img_p)  # child

            father_img_paths.append(father_img_p)
            mother_img_paths.append(mother_img_p)
            child_img_paths.append(child_img_p)

    # father_img_paths_orig = father_img_paths.copy()
    # father_img_paths_orig = father_img_paths.copy()
    # child_img_paths_orig = child_img_paths.copy()
    n = len(father_img_paths)

    # the plotting code
    p1 = figure(height=300, width=300)
    source = ColumnDataSource(data=dict(url=[father_img_paths[0]] * n,
                                        url_orig=father_img_paths,
                                        x=[1] * n, y=[1] * n, w=[1] * n, h=[1] * n))
    image1 = ImageURL(url="url", x="x", y="y", w="w", h="h", anchor="bottom_left")
    p1.add_glyph(source, glyph=image1)

    p2 = figure(height=300, width=300)
    source2 = ColumnDataSource(data=dict(url=[mother_img_paths[0]] * n,
                                         url_orig=mother_img_paths,
                                         x=[1] * n, y=[1] * n, w=[1] * n, h=[1] * n))
    image2 = ImageURL(url="url", x="x", y="y", w="w", h="h", anchor="bottom_left")
    p2.add_glyph(source2, glyph=image2)
    
    p3 = figure(height=300, width=300)
    source3 = ColumnDataSource(data=dict(url=[child_img_paths[0]] * n,
                                         url_orig=child_img_paths,
                                         x=[1] * n, y=[1] * n, w=[1] * n, h=[1] * n))
    image3 = ImageURL(url="url", x="x", y="y", w="w", h="h", anchor="bottom_left")
    p3.add_glyph(source2, glyph=image3)
    

    # the callback
    callback = CustomJS(args=dict(source=source, source2=source2, source3=source3), code="""
        var f = cb_obj.value;

        var data = source.data;    
        url = data['url']
        url_orig = data['url_orig']
        console.log(url)
        console.log(url_orig)
        for (i = 0; i < url_orig.length; i++) {
            url[i] = url_orig[f-1]
        }
        source.change.emit();


        var data = source2.data;    
        url = data['url']
        url_orig = data['url_orig']
        console.log(url)
        console.log(url_orig)
        for (i = 0; i < url_orig.length; i++) {
            url[i] = url_orig[f-1]
        }
        source2.change.emit();
        
        
        var data = source3.data;    
        url = data['url']
        url_orig = data['url_orig']
        console.log(url)
        console.log(url_orig)
        for (i = 0; i < url_orig.length; i++) {
            url[i] = url_orig[f-1]
        }
        source3.change.emit();

    """)
    slider = Slider(start=1, end=n, value=1, step=1, title="example number")
    slider.js_on_change('value', callback)

    layout = column(slider, row(p1, p2, p3))

    show(layout)


def _save_pkl_images_to_local_path(pkl_folder_path) -> (list, list, list):
    """
    :param pkl_folder_path: path with triplet pickle files
    :return: Returns (father_img_paths, mother_img_paths, child_img_paths) with local paths for bokeh
    """
    # Save folders in curr folder for bokeh access
    os.makedirs("pics/", exist_ok=True)
    father_img_paths, mother_img_paths, child_img_paths = [], [], []
    for i, filep in enumerate(tqdm(get_files_from_path(pkl_folder_path))):
        with open(filep, 'rb') as f:
            (father_image, father_latent_f), (mother_image, mother_latent_f), (child_image, child_latent_f) = pkl.load(
                f)

            father_img_p, mother_img_p, child_img_p = f'pics/{i}-F.png', f'pics/{i}-M.png', f'pics/{i}-C.png'

            father_image.save(father_img_p)
            mother_image.save(mother_img_p)
            child_image.save(child_img_p)  # child

            father_img_paths.append(father_img_p)
            mother_img_paths.append(mother_img_p)
            child_img_paths.append(child_img_p)

    return father_img_paths, mother_img_paths, child_img_paths


def _save_pred_images_to_local_path(pkl_folder_path, predictions_folder_path) -> list:
    """
    :param pkl_folder_path: path with models prediction folders
    :return: Returns (father_img_paths, mother_img_paths, child_img_paths) with local paths for bokeh
    """
    model_names = [o for o in os.listdir(predictions_folder_path) if os.path.isdir(os.path.join(predictions_folder_path, o))]

    models_predictions_path_list = [[]] * len(model_names)

    # Save folders in curr folder for bokeh access
    os.makedirs("pics/", exist_ok=True)
    for i, filep in enumerate(get_files_from_path(pkl_folder_path)):
        fname = os.path.basename(filep)

        for j, model_name in enumerate(model_names):
            model_ex_path = f"{predictions_folder_path}/{model_name}/{fname}"
            if os.path.exists(model_ex_path):
                with open(model_ex_path, 'rb') as f:
                    (model_pred_child_img, model_pred_child_latent_f) = pkl.load(f)
            else:
                model_pred_child_img = Image.fromarray(np.zeros((3, 1024, 1024)), 'RGB')

            model_pred_local_p = f'pics/{i}-pred-{model_name}.png'
            model_pred_child_img.save(model_pred_local_p)
            models_predictions_path_list[j].append(model_pred_local_p)

    return models_predictions_path_list



def family_view_with_slider_upgraded(pkl_folder_path):
    # TODO: not tested

    father_img_paths, mother_img_paths, child_img_paths = _save_pkl_images_to_local_path(pkl_folder_path)
    n = len(father_img_paths)

    # the plotting code
    plots = []
    sources = []
    pathes = [father_img_paths, mother_img_paths, child_img_paths]
    plot_num = 3
    
    for i in range(plot_num):
        p = figure(height=300, width=300)
        img_paths = pathes[i]
        # print(img_paths)
        source = ColumnDataSource(data=dict(url=[img_paths[0]] * n,
                                            url_orig=img_paths,
                                            x=[1] * n, y=[1] * n, w=[1] * n, h=[1] * n))
        image = ImageURL(url="url", x="x", y="y", w="w", h="h", anchor="bottom_left")
        p.add_glyph(source, glyph=image)
        p.toolbar.logo = None
        p.toolbar_location = None
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None

        p.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
        p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
        p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
        p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
        p.xaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels
        p.yaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels

        plots.append(p)
        sources.append(source)


    update_source_str = """
    
        var data = source{i}.data;    
        url = data['url']
        url_orig = data['url_orig']
        console.log(url)
        console.log(url_orig)
        for (i = 0; i < url_orig.length; i++) {
            url[i] = url_orig[f-1]
        }
        source{i}.change.emit();
        
    """
    # the callback
    callback = CustomJS(args=dict(source0=sources[0], source1=sources[1], source2=sources[2]), code=f"""
        var f = cb_obj.value;
        
        {"".join([update_source_str.replace('{i}', str(i)) for i in range(plot_num)])}
    """)
    slider = Slider(start=1, end=n, value=1, step=1, title="example number")
    slider.js_on_change('value', callback)

    layout = column(slider, row(*plots))

    show(layout)


def family_view_with_slider_and_predictions(pkl_folder_path, predictions_folder_path):
    # TODO: not tested

    father_img_paths, mother_img_paths, child_img_paths = _save_pkl_images_to_local_path(pkl_folder_path)
    n = len(father_img_paths)

    # the plotting code

    sources_dict = {}

    # parent plots
    family_plots = []

    pathes = [father_img_paths, mother_img_paths, child_img_paths]
    family_plot_num = 3

    for i in range(family_plot_num):

        p = figure(height=300, width=300)
        img_paths = pathes[i]
        source = ColumnDataSource(data=dict(url=[img_paths[0]] * n,
                                            url_orig=img_paths,
                                            x=[1] * n, y=[1] * n, w=[1] * n, h=[1] * n))
        image = ImageURL(url="url", x="x", y="y", w="w", h="h", anchor="bottom_left")
        p.add_glyph(source, glyph=image)
        p.toolbar.logo = None
        p.toolbar_location = None
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None

        p.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
        p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
        p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
        p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
        p.xaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels
        p.yaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels

        family_plots.append(p)
        sources_dict[f"source{i}"] = source

    # model prediction plots
    pred_plots = []
    model_names = [o for o in os.listdir(predictions_folder_path) if os.path.isdir(os.path.join(predictions_folder_path, o))]

    pathes = _save_pred_images_to_local_path(pkl_folder_path, predictions_folder_path)
    pred_plot_num = len(model_names)

    for i, model_name in enumerate(model_names):
        p = figure(height=300, width=300, title=model_name)
        img_paths = pathes[i]
        source = ColumnDataSource(data=dict(url=[img_paths[0]] * n,
                                            url_orig=img_paths,
                                            x=[1] * n, y=[1] * n, w=[1] * n, h=[1] * n))
        image = ImageURL(url="url", x="x", y="y", w="w", h="h", anchor="bottom_left")
        p.add_glyph(source, glyph=image)

        p.toolbar.logo = None
        p.toolbar_location = None
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None

        p.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
        p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
        p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
        p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
        p.xaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels
        p.yaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels

        pred_plots.append(p)
        sources_dict[f"source{family_plot_num+i}"] = source

    update_source_str = """

        var data = source{i}.data;    
        url = data['url']
        url_orig = data['url_orig']
        console.log(url)
        console.log(url_orig)
        for (i = 0; i < url_orig.length; i++) {
            url[i] = url_orig[f-1]
        }
        source{i}.change.emit();

    """
    # the callback
    callback = CustomJS(args=sources_dict, code=f"""
        var f = cb_obj.value;

        {"".join([update_source_str.replace('{i}', str(i)) for i in range(family_plot_num+pred_plot_num)])}
    """)
    slider = Slider(start=1, end=n, value=1, step=1, title="example number")
    slider.js_on_change('value', callback)

    layout = column(slider, row(*family_plots), row(*pred_plots))

    show(layout)
