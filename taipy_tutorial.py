from taipy.gui import Gui
import taipy.gui.builder as tgb

if __name__ == '__main__':
    text = 'Enter the name'
    Slider = 0

    with tgb.Page() as page:
        tgb.text('# Getting started with *Taipy* Gui', mode='md')
        tgb.slider('{Slider}', min=90, max=180)
        tgb.text('My name is: {text}')
        tgb.input('{text}')

    Gui(page).run(debug=True, use_reloader=True)

