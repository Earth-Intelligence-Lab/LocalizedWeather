import cartopy
import cartopy.crs as ccrs
import numpy as np
from matplotlib import pyplot as plt


def PlotStations(madis_network, era5_network, output_path):

    if era5_network is not None:
        era_5_coord = era5_network.era5_pos.cpu().numpy()
    else:
        era_5_coord = None


    edge = madis_network.k_edge_index.long().cpu().numpy()

    madis_lat = madis_network.madis_lat.cpu().numpy()
    madis_lon = madis_network.madis_lon.cpu().numpy()

    if era5_network is not None:
        e2m_edge_index = era5_network.e2m_edge_index.cpu().numpy()
    else:
        e2m_edge_index = None


    __PlotStations(
                    madis_lat,
                    madis_lon,
                    edge,
                    era_5_coord,
                    e2m_edge_index,
                    output_path)
    

def __PlotStations(
                    madis_lat,
                    madis_lon,
                    edges,
                    era_5_coord,
                    e2m_edge_index,
                    output_path):


    fig, ax = plt.subplots(1, 1, figsize=(8, 5), layout='tight', dpi=300, subplot_kw=dict(projection=ccrs.PlateCarree()))

    bodr = cartopy.feature.NaturalEarthFeature(category='cultural',
                                               name='admin_1_states_provinces', scale='10m')
    ax.add_feature(bodr, edgecolor="k", lw=1, facecolor='lightgrey')

    madis_node_label = 'Madis Node'
    bi_line_label = 'Bi-directional edge'
    uni_line_label = 'Uni-directional edge'
    era5__line_label = 'ERA5 Edge'
    era5__point_label = 'ERA 5 Node'

    if era_5_coord is not None:
        era5_coords_set = era_5_coord[e2m_edge_index[0, :]]
        madis_lat_set = madis_lat[e2m_edge_index[1, :]].squeeze()
        madis_lon_set = madis_lon[e2m_edge_index[1, :]].squeeze()

        for i in range(len(era5_coords_set)):
            ax.plot([era5_coords_set[i, 0], madis_lon_set[i]],
                        [era5_coords_set[i, 1], madis_lat_set[i]], color='seagreen', lw=.5, alpha=.5, label=era5__line_label)


        era5_points_set = np.unique(e2m_edge_index[0,:])
        era5_coords_set = era_5_coord[era5_points_set]

        ax.plot(era5_coords_set[:, 0], era5_coords_set[:, 1], 'o', color='seagreen', alpha=.5, label=era5__point_label)



    nb_edges = len(edges[0])

    longitudes_i = madis_lon[edges[0]]
    longitudes_j = madis_lon[edges[1]]

    latitudes_i = madis_lat[edges[0]]
    latitudes_j = madis_lat[edges[1]]

    for edge_i in range(nb_edges):
        edge = edges[:, edge_i]
        arrowcolor = 'salmon'
        arrowlabel = uni_line_label
        if not np.any(np.sum(edges.T == np.flip(edge), axis=1) == 2):
            dx = (longitudes_j[edge_i] - longitudes_i[edge_i])[0]
            dy = (latitudes_j[edge_i] - latitudes_i[edge_i])[0]

            norm = np.array([dx, dy]) / np.sqrt(dx ** 2 + dy ** 2)

            offset = norm * .18

            dx = dx - 2 * offset[0]
            dy = dy - 2 * offset[1]

            ax.arrow(longitudes_i[edge_i][0] + offset[0], latitudes_i[edge_i][0] + offset[1], dx,
                             dy, head_width=.02, color=arrowcolor, zorder=100, label=arrowlabel,
                             length_includes_head=True)
        else:
            ax.plot([longitudes_i[edge_i][0], longitudes_j[edge_i][0]],
                             [latitudes_i[edge_i][0], latitudes_j[edge_i][0]], color='k', label=bi_line_label)

    ax.plot(madis_lon, madis_lat, 'o', color='w', ms=8, mec='k', zorder=200, label=madis_node_label)

    handles, label = ax.get_legend_handles_labels()

    handles_display = []

    ind = np.where(np.array(label) == madis_node_label)[0][0]
    handles_display.append(handles[ind])
    ind = np.where(np.array(label) == uni_line_label)[0][0]
    handles_display.append(handles[ind])
    ind = np.where(np.array(label) == bi_line_label)[0][0]
    handles_display.append(handles[ind])
    ind = np.where(np.array(label) == era5__line_label)[0][0]
    handles_display.append(handles[ind])
    ind = np.where(np.array(label) == era5__point_label)[0][0]
    handles_display.append(handles[ind])

    ax.legend(handles=handles_display, loc='best', fontsize=12)

    fig.savefig(output_path/f'stationSetup_plot.png', bbox_inches='tight')
    fig.savefig(output_path/f'stationSetup_plot.pdf', bbox_inches='tight')
    plt.close(fig)