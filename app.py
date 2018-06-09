from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
import requests
import bokeh
import holoviews as hv
import pandas as pd
app = Flask(__name__)