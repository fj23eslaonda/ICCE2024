#------------------------------------------------------------------------------- 
#
# PACKAGES
#
#------------------------------------------------------------------------------- 
import ipywidgets as widgets
import time
from jupyter_ui_poll import ui_events
import matplotlib.pyplot as plt
from IPython.display import clear_output

#------------------------------------------------------------------------------- 
#
# IMAGES QUERY
#
#------------------------------------------------------------------------------- 
def images_query(x_tr, y_tr):
    
  # GLOBAL VARIABLES
  global ui_done
  global score_per_image
  
  score_per_image = list()

  for i in range(len(x_tr)):
    clear_output(wait=True)
    ui_done = False

    print()
    print()

    progress = widgets.IntProgress(
        value=i,
        min=0,
        max=len(x_tr),
        description='Loading:',
        bar_style='success', # 'success', 'info', 'warning', 'danger' or ''
        style={'bar_color': 'green'},
        orientation='horizontal',
        layout=widgets.Layout(width='30%', height='25px'))
    
    display(progress)

    # TITLE
    title = widgets.HTML(value = "<b><center><font size=4.5>Classification of images according to expert criteria</b></center>")
    display(title)

    # IMAGES PLOT
    plot_images_query(x_tr,y_tr, i)

    print()
    # INSTRUCTIONS
    query = widgets.HTML(value = "<b><center><font size=4.5>Classify the image from 1 to 5 according to net\'s performance</b></center>") 
    display(query)

    print()

    # BUTTONS AND TEXT
    txt  = widgets.HTML(value = "<b><font size=3>         Enter your score: </b>") 
    
    btn1 = widgets.Button(description='1')
    btn2 = widgets.Button(description='2')
    btn3 = widgets.Button(description='3')
    btn4 = widgets.Button(description='4')
    btn5 = widgets.Button(description='5')

    btn1.on_click(on_click)
    btn2.on_click(on_click)
    btn3.on_click(on_click)
    btn4.on_click(on_click)
    btn5.on_click(on_click)

    # BUTTONS PLOT
    display(widgets.HBox([txt,btn1,btn2,btn3,btn4,btn5], layout=widgets.Layout(display='flex',
                align_items='center',
                width='100%')))

    # WAIT FOR USERS TO PRESS THE BUTTON
    with ui_events() as poll:
        while ui_done is False:
            poll(10)
            time.sleep(0.1)

    print()
    print()

  return score_per_image

#------------------------------------------------------------------------------- 
#
# CLICK FUNCTION
#
#------------------------------------------------------------------------------- 
def on_click(btn):

  global ui_done
  global score_per_image

  ui_done = True
  score_per_image.append(float(int(btn.description)))


#------------------------------------------------------------------------------- 
#
# PLOT IMAGES
#
#------------------------------------------------------------------------------- 

def plot_images_query(x_tr, y_tr, index):

  # SETTING                     
  fig, ax = plt.subplots(1, 3, figsize=(14, 5))

  fig.subplots_adjust(left=-0.5,
                      right=1,
                      bottom=0,
                      top=0.8,
                      wspace=-0.1,
                      hspace=-1
                      )
  # IMAGE NUMBER
  ix = index

  # FIGURE 1
  ax[0].imshow(x_tr[ix].squeeze(), cmap='gray')
  ax[0].grid(False)
  ax[0].set_title('Video image', fontsize=13)
  ax[0].set_xlabel('Cross-shore distance, x [pixels]', fontsize=13)
  ax[0].set_ylabel('Alongshore distance, y [pixels]', fontsize=13)

  # FIGURE 2
  ax[1].imshow(y_tr[ix].squeeze(), cmap='gray')
  ax[1].grid(False)
  ax[1].set_title('Prediction Binary mask', fontsize=13)
  ax[1].set_xlabel('Cross-shore distance, x [pixels]', fontsize=13)
  ax[1].set_ylabel('Alongshore distance, y [pixels]', fontsize=13)

  # FIGURE 3
  ax[2].imshow(x_tr[ix].squeeze(), cmap='gray')
  ax[2].contour(y_tr[ix].squeeze(), colors='r', levels=[0.1])
  ax[2].grid(False)
  ax[2].set_title('Video Image and Binary Mask', fontsize=13)
  ax[2].set_xlabel('Cross-shore distance, x [pixels]', fontsize=13)
  ax[2].set_ylabel('Alongshore distance, y [pixels]', fontsize=13)

  plt.show()
  plt.close(fig)  # Close the figure to free up memory
