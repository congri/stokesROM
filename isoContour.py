
import numpy as np

class IsoContour:
    Objects = []    # distinct objects
    Vertices = []   # Vertices of isocontour


    def isocontour(self, img, isovalue = 0.0):
        """
        #  function [Lines,Vertices,Objects]=isocontour(I,isovalue)
        #  This function ISOCONTOUR computes the isocontour geometry for
        #  a certain 2D image and isovalue. To Extract the isocontour geometry it
        #  uses Marching Squares and linear interpolation. Followed by sorting the
        #  contour geometry into separate sorted contours.
        #
        #  This function is the 2D equivalent of Isosurface extraction
        #  using Marching Cubes in 3D.
        #
        #
        #    [Lines,Vertices,Objects]=isocontour(I,isovalue);
        #
        #  inputs,
        #    I : An 2D image (grey-scale)
        #    isovalue : The Iso-value of the contour
        #
        #  outputs,
        #    Lines : An array describing all the Line-pieces of the isocontour
        #            geomtery, with a N x 2 index list of vertices
        #    Vertices : Vertices (Corners) of the lines M x 2 list of X,Y
        #            coordinates
        #    Objects : A 1 x K  cell array with in every cell a list of indices
        #            corresponding to one connected isocontour. If the isocontour
        #            is closed then the last index value is equal to first index
        #            value.
        #
        #  Note : This function displays the image with isocontours if no output
        #        is defined.
        #
        #  Example,
        #     I = im2double(imread('rice.png'));
        #     isocontour(I,0.5);
        #
        #  Example,
        #     I = im2double(imread('rice.png'));
        #     [Lines,Vertices,Objects]=isocontour(I,0.5);
        #     figure('renderer','opengl'), imshow(I), hold on;
        #     for i=1:length(Objects)
        #          Points=Objects{i};
        #          plot(Vertices(Points,2),Vertices(Points,1),'Color',rand(3,1));
        #     end
        #
        #  Example,
        #     I = im2double(imread('rice.png'));
        #     [Lines,Vertices]=isocontour(I,0.5);
        #     figure, imshow(I), hold on;
        #     V1=Vertices(Lines(:,1),:); V2=Vertices(Lines(:,2),:);
        #     plot([V1(:,2) V2(:,2)]',[V1(:,1) V2(:,1)]','b');
        #     #  Calculate Contour Normals
        #     N = V1-V2; L = sqrt(N(:,1).^2+N(:,2).^2)+eps;
        #     N(:,1)=N(:,1)./L; N(:,2)=-N(:,2)./L;
        #     plot([V1(:,2) V1(:,2)+N(:,1)*5]',[V1(:,1) V1(:,1)+N(:,2)*5]','r');
        #
        #
        """


        #  Check Inputs
        # if(nargin==0), error('isocontour:input','no input image defined'); end
        if img.ndim != 2:
            raise Exception('isocontour:input' ,'image must be 2D')

        # function[V, F] = LookupDB(Img, isovalue)
        def lookupDB(img2, isovalue):
            #  Describe the base-polygons by edges
            #   Edge number
            #    1
            #   ***
            #  0* *2
            #   ***
            #    3
            #
            I = 16* [[0, 0, 0, 0]]
            I = np.array(I)
            I[0, :] = [4, 4, 4, 4]  # [0 0;0 0]
            I[1, :] = [1, 0, 4, 4]  # [1 0;0 0]
            I[2, :] = [2, 1, 4, 4]  # [0 1;0 0]
            I[3, :] = [2, 0, 4, 4]  # [1 1;0 0]
            I[4, :] = [0, 3, 4, 4]  # [0 0;1 0]
            I[5, :] = [1, 3, 4, 4]  # [1 0;1 0]
            I[6, :] = [2, 1, 0, 3]  # [0 1;1 0] ambiguous
            I[7, :] = [2, 3, 4, 4]  # [1 1;1 0]
            I[8, :] = [3, 2, 4, 4]  # [0 0;0 1]
            I[9, :] = [1, 0, 3, 2]  # [1 0;0 1] ambiguous
            I[10, :] = [3, 1, 4, 4]  # [0 1;0 1]
            I[11, :] = [3, 0, 4, 4]  # [1 1;0 1]
            I[12, :] = [0, 2, 4, 4]  # [0 0;1 1]
            I[13, :] = [1, 2, 4, 4]  # [1 0;1 1]
            I[14, :] = [0, 1, 4, 4]  # [0 1;1 1]
            I[15, :] = [4, 4, 4, 4]  # [1 1;1 1]

            #  The base-edges by vertex positions
            E = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 1, 1], [1, 0, 1, 1], [4, 4, 4, 4]])

            #  base-Polygons by vertexpostions
            Iflat = np.ndarray.flatten(I, 'F')
            IE = E[Iflat, :]
            IE = np.concatenate((np.concatenate((IE[0:16, :], IE[16:32, :]), axis=1),
                                 np.concatenate((IE[32:48, :], IE[48:64, :]), axis=1)), axis=0)

            #  Make a Binary image with pixels set to true above iso-treshold
            B = img2 >= isovalue

            #  Get Elementary Cells
            #   Cell
            #   1 ** 2
            #   *    *
            #   *    *
            #   4 ** 8
            #
            B0 = B[0:-1, 0:-1]
            B1 = B[0:-1, 1:]
            B2 = B[1:, 0:-1]
            B3 = B[1:, 1:]
            V = B0 + 2 * B1 + 4 * B2 + 8 * B3 + 1

            x = np.argwhere(np.transpose(np.logical_and(V > 1, V < 16)))
            y = x[:, 0]
            x = x[:, 1]
            Vflat = np.ndarray.flatten(V, 'F')
            v = Vflat[x + y*V.shape[0]] - 1

            #  Elementary cells to Edge coordinates defined by connected image grid-points
            J = np.concatenate((IE[v, :], IE[v + 16, :]), axis=0)
            r = (J[:, 0] == 4)
            x = np.expand_dims(x, axis=1)
            y = np.expand_dims(y, axis=1)
            xy = np.concatenate((np.concatenate((x, y, x, y, x, y, x, y), axis=1),
                                 np.concatenate((x, y, x, y, x, y, x, y), axis=1)), axis=0)
            J = J + xy
            r = np.logical_not(r)
            J = J[r, :]

            #  Vertices list defined by connected image grid-points
            VP = np.concatenate((J[:, 0:4], J[:, 4:8]), axis=0)

            #  Make a Face list
            F = np.arange(0, 2*J.shape[0])
            F = np.transpose(np.reshape(F, [2, J.shape[0]]))

            #  Remove dubplicate vertices
            VP, Ind = np.unique(VP, return_inverse=True, axis=0)
            F = Ind[F]

            #  Vertices described by image grid-points to real
            #  linear Interpolated vertices
            Vind1 = VP[:, 0] + VP[:, 1]*img2.shape[0]
            Vind2 = VP[:, 2] + VP[:, 3]*img2.shape[0]
            img2flat = np.ndarray.flatten(img2, 'F')
            V1 = abs(img2flat[Vind1] - isovalue)
            V2 = abs(img2flat[Vind2] - isovalue)
            alpha = V2 / (V1 + V2)
            Vx = VP[:, 0]*alpha + VP[:, 2]*(1 - alpha) + 1
            Vy = VP[:, 1]*alpha + VP[:, 3]*(1 - alpha) + 1
            Vx = np.expand_dims(Vx, axis=1)
            Vy = np.expand_dims(Vy, axis=1)
            V = np.concatenate((Vx, Vy), axis=1)

            return V, F

        # function Objects=SortLines2Objects(Lines)
        def sortLines2Objects(lines):
            #  Object index list
            Obj = np.zeros([512, 2], dtype=int)
            Obj[0, 0] = 0
            nObjects = 1
            reverse = False
            for i in range(0, lines.shape[0] - 1):
                F = lines[i, 1]
                Q = i + 1
                #print(i)
                #print(lines[0:5, :])
                if np.max(np.any(lines[Q:, :] == F, axis=1)):
                    R = np.argmax(np.any(lines[Q:, :] == F, axis=1))
                else:
                    R = np.array([])
                if R.size:
                    R = R + i + 1
                    TF = lines[Q, :].copy()
                    if lines[R, 0] == F:
                        lines[Q, :] = lines[R, :]
                    else:
                        lines[Q, :] = lines[R, [1, 0]]

                    if R != Q:
                        lines[R, :] = TF
                else:
                    F = lines[Obj[nObjects - 1, 0], 0]
                    if np.max(np.any(lines[Q:, :] == F, axis=1)):
                        R = np.argmax(np.any(lines[Q:, :] == F, axis=1))
                    else:
                        R = np.array([])
                    if R.size:
                        reverse = True
                        lines[Obj[nObjects - 1, 0]:(i + 1), :] = \
                            np.flip(lines[Obj[nObjects - 1, 0]:(i + 1), [1, 0]], axis=0)
                        R = R + i + 1
                        TF = lines[Q, :].copy()
                        if lines[R, 0] == F:
                            lines[Q, :] = lines[R, [0, 1]]
                        else:
                            lines[Q, :] = lines[R, [1, 0]]
                        if R != Q:
                            lines[R, :] = TF
                    else:
                        if reverse:
                            lines[Obj[nObjects - 1, 0]:(i + 1), :] =\
                                np.flip(lines[Obj[nObjects - 1, 0]:(i + 1), [1, 0]], axis=0)
                            reverse = False
                        Obj[nObjects - 1, 1] = i
                        nObjects += 1
                        Obj[nObjects - 1, 0] = i + 1
            Obj[nObjects - 1, 1] = i + 1
            #  Object index list, to real connect object lines
            Objects = nObjects*[None]
            for i in range(0, nObjects):
                #  Determine if the line is closed
                if lines[Obj[i, 0], 0] == lines[Obj[i, 1], 1]:
                    Objects[i] = np.append(lines[Obj[i, 0]:(Obj[i, 1]), 0], lines[Obj[i, 0], 0])
                else:
                    temp = np.expand_dims(np.arange(Obj[i, 0], Obj[i, 1] + 1), axis=1)
                    Objects[i] = lines[temp, 0]
            return Objects

        #  Get the Line Pieces
        [Vertices, Lines] = lookupDB(img, isovalue)

        #  Sort the Line Pieces into objects
        Objects = sortLines2Objects(Lines)

        #  Show image
        showImg = True
        if showImg:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cax = ax.imshow(img >= isovalue, cmap=plt.cm.inferno)
            for i in range(0, len(Objects)):
                Points = Objects[i]
                plt.plot(Vertices[Points, 1] - 1, Vertices[Points, 0] - 1)
            plt.show(block=False)


        print('Vertices = ', Vertices)
        # Normalize vertices
        Vertices[:, 0] = (Vertices[:, 0] - 1.0)/(img.shape[0] - 1.0)
        Vertices[:, 1] = (Vertices[:, 1] - 1.0)/(img.shape[1] - 1.0)
        print('Normalized Vertices = ', Vertices)
        self.Objects = Objects
        self.Vertices = Vertices
        return Objects, Vertices, Lines
