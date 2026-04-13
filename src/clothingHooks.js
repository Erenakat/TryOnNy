/**
 * Clothing roadmap – hooks for future try-on.
 * Option A: 3D garments (GLTF) skinned to the avatar rig.
 * Option B: Cloth simulation / size recommendations (later).
 * MVP: avatar only; this file exposes hooks for garment attachment.
 */

/**
 * @param {string} avatarGlbUrl – URL to the current avatar GLB
 * @param {string} [garmentId] – future: id of garment to attach (e.g. 'tshirt', 'pants')
 * @returns {string|null} – future: URL to composed avatar+garment GLB or overlay config; MVP returns null
 */
export function getGarmentAttachmentUrl(avatarGlbUrl, garmentId) {
  if (!garmentId) return null;
  // TODO Option A: backend endpoint that returns composed GLB (avatar + skinned garment)
  // e.g. GET /avatar/outfit?avatarUrl=...&garment=tshirt
  return null;
}

/**
 * @param {string} avatarGlbUrl
 * @param {string} garmentId
 * @returns {Promise<{ success: boolean, composedUrl?: string, error?: string }>}
 */
export async function attachGarment(avatarGlbUrl, garmentId) {
  // TODO: call backend to attach 3D garment to avatar rig; return composed GLB URL
  return { success: false, error: 'Not implemented' };
}
