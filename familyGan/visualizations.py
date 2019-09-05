import os

from bokeh.plotting import figure, show, output_file

from bokeh.models.glyphs import ImageURL

from bokeh.models import Image, ColumnDataSource
from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models import CustomJS, ColumnDataSource, Slider
from bokeh.plotting import Figure, show, figure
from bokeh.layouts import row
from PIL import Image
import numpy as np
import pickle as pkl
from bokeh.plotting import figure
# graph.write_png("dtree.png")
# from bokeh import Figure
# wheel_zoom = WheelZoomTool()
from bokeh.models.tools import WheelZoomTool

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
