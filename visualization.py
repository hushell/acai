import os
import torch
import torchvision.transforms as transforms
import numpy as np


def images_to_grid(images):
	"""Converts a grid of images (5D tensor) to a single image.

	Args:
	images: 5D tensor (count_y, count_x, channels, height, width), grid of images.

	Returns:
	3D tensor of shape (channels, count_y * height, count_x * width).
	"""
	ny, nx, c, h, w = images.size()
	image = images.permute(2, 0, 3, 1, 4)
	image = torch.reshape(image, (c, ny * h, nx * w))
	return image


def to_png(x):
	"""Convert a 3D tensor to png."""
	return torch.clamp(torch.round(x * 127.5 + 127.5), 0, 255)


def save_image(image, image_dir, name):
	"""Convert a 3D tensor to PIL Image and save it."""
	to_PIL = transforms.ToPILImage('RGB')
	image = to_PIL(image)
	image.save(os.path.join(image_dir, name))


def make_sample_grid_and_save(encoder_x, decoder_mix, data_loader, image_dir=None, batch_size=16, interpolation=16):
	"""
	Args:
	encoder_x (nn.module): Encoder for target domain images.
	decoder_mix (nn.module): Decoder for mixed domain images.
	data_loader: Data loader for loading images from two domains.
	image_dir (str): Directory to save the generated images.
	batch_size (int): Number of samples to interpolate at each time.
	interpolation (int): Number of interpolations for each sample.

	Returns:
	images of shape HxWxC with values in [0, 255].
	"""

	encoder_x.eval()
	decoder_mix.eval()

	with torch.no_grad():
		# Gather photos of size NxCxHxW
		photos, _ = next(iter(data_loader))
		photos = photos.cuda() # TODO: to(device)
		if batch_size <= photos.size(0):
			photos = photos[:batch_size]
		else:
			raise ValueError('Data loader should have a larger batch_size.')

		# Encode the photos and sketches to get the latent vectors of size NxD
		latents_x = encoder_x(photos)
		Nx, Cx, Wx, Hx = latents_x.size()
		latents_x = latents_x.view(Nx, -1)
		latents_s = latents_x.flip(0)
		Ns, Cs, Ws, Hs = Nx, Cx, Wx, Hx


		# Interpolate the latent vectors (lerp and slerp)
		latents_lerp = torch.zeros((batch_size * interpolation, latents_x.size(1)), dtype=torch.float).cuda()
		latents_slerp = torch.zeros((batch_size * interpolation, latents_x.size(1)), dtype=torch.float).cuda()
		dots = torch.sum(latents_x * latents_s, 1)
		norms = torch.sqrt(torch.sum(latents_x ** 2, 1)) * torch.sqrt(torch.sum(latents_s ** 2, 1))
		omegas = torch.acos(dots / norms).view(-1, 1).expand_as(latents_x)
		for i in range(interpolation):
			latents_lerp[i * batch_size:(i + 1) * batch_size] = (latents_s * (interpolation - 1 - i) + latents_x * i) / float(interpolation - 1)
			t = i / (interpolation - 1)
			latents_slerp[i * batch_size:(i + 1) * batch_size] = (torch.sin((1 - t) * omegas) * latents_s + torch.sin(t * omegas) * latents_x) / torch.sin(omegas)

		# Decode the interpolated latent vectors into a grid of images of shape CxHxW
		images_lerp = decoder_mix(latents_lerp.view(Nx * interpolation, Cx, Wx, Hx))
		images_lerp = images_lerp.view(interpolation, batch_size, images_lerp.size(1), images_lerp.size(2), images_lerp.size(3)).transpose(0, 1)
		images_slerp = decoder_mix(latents_slerp.view(Ns * interpolation, Cs, Ws, Hs))
		images_slerp = images_slerp.view(interpolation, batch_size, images_slerp.size(1), images_slerp.size(2), images_slerp.size(3)).transpose(0, 1)

		# Concatenate the sketches at the first column and the photos at the last column
		images_lerp = torch.cat((images_lerp, photos.unsqueeze(1)), 1)
		images_slerp = torch.cat((images_slerp, photos.unsqueeze(1)), 1)

		# Convert the grid of images to a single image with value in [0, 255]
		image_lerp = to_png(images_to_grid(images_lerp))
		image_slerp = to_png(images_to_grid(images_slerp))

		# Save PIL image to image_dir
		if image_dir is not None:
			save_image(image_lerp, image_dir, 'linear_interpolation.png')
			save_image(image_slerp, image_dir, 'sphere_interpolation.png')

		encoder_x.train()
		decoder_mix.train()

		return image_lerp.cpu().numpy().astype(np.uint8), image_slerp.cpu().numpy().astype(np.uint8)


if __name__ == "__main__":
	def encode(x):
		return x.view(x.size(0), -1)

	def decode(x):
		return x.view(x.size(0), 3, 32, 32)

	from dataloader import TrainValDataLoader
	test_loader = TrainValDataLoader('/space_sde/ShoeV2_F', train=False)
	im_lerp, im_slerp = make_sample_grid_and_save(encode, decode, test_loader, ".")
	print(im_lerp.size(), im_slerp.size())
